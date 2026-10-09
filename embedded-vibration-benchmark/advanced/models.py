import torch
from torch import nn

INPUTS, HIDDEN = 6, 16
BETA, RHO, BASE_THRESHOLD, ADAPT_STRENGTH = .875, .95, 1., .4


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (voltage,) = ctx.saved_tensors
        return gradient / (1 + 5 * voltage.abs()).pow(2)


class TemporalModel(nn.Module):
    """Capacity-matched trainable 6->16->1 LIF, ALIF, or mean-ReLU ANN.

    Every window resets state. Input is continuous nonnegative current, not binary
    input spikes. Hidden spikes are hard in forward and surrogate in backward.
    """
    def __init__(self, kind):
        super().__init__()
        if kind not in ('lif', 'alif', 'ann'):
            raise ValueError(kind)
        self.kind = kind
        self.input_layer = nn.Linear(INPUTS, HIDDEN)
        self.output_layer = nn.Linear(HIDDEN, 1)
        with torch.no_grad():
            self.input_layer.weight.normal_(.20, .20)
            self.input_layer.bias.fill_(.12)

    def forward(self, inputs, return_state=False):
        voltage = inputs.new_zeros((len(inputs), HIDDEN))
        adaptation = torch.zeros_like(voltage)
        counts = torch.zeros_like(voltage)
        for tick in range(inputs.shape[1]):
            current = self.input_layer(inputs[:, tick])
            if self.kind == 'ann':
                activity = torch.relu(current)
            else:
                voltage = BETA * voltage + current
                threshold = BASE_THRESHOLD + (ADAPT_STRENGTH * adaptation if self.kind == 'alif' else 0)
                activity = SurrogateSpike.apply(voltage - threshold)
                # Detaching reset spikes is a training convention only.
                voltage = voltage - activity.detach() * threshold
                if self.kind == 'alif':
                    adaptation = RHO * adaptation + activity
            counts = counts + activity
        logit = self.output_layer(counts / inputs.shape[1]).squeeze(1)
        if return_state:
            return logit, voltage, adaptation, counts
        return logit


class LogisticControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = nn.Linear(INPUTS, 1)

    def forward(self, inputs):
        return self.readout(inputs.mean(dim=1)).squeeze(1)
