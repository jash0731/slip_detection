# Understand the project and explain it in the viva

## The project in everyday language

Imagine a robot holding a cup. It needs to notice when the cup starts slipping.
Contact sensors produce changing signals. Our project asks whether a small
brain-inspired neural network can interpret those signals on an inexpensive
ESP32, and what it costs compared with an ordinary neural network.

We use recorded signals from real robotic-gripper experiments. That lets us
study the computing problem without building a gripper. The laptop trains and
evaluates models. We export their learned numbers into Arduino C++, and the
ESP32 runs inference on a recorded input sequence. A motor is not involved.

The final trained model does not automatically understand your vibration sensor.
The public recordings use a different tactile sensor. A vibration-to-LED demo
would demonstrate acquisition and an alarm, with its own settings and limits.

## What the parts mean

- **ESP32:** the small computer that executes the learned detector.
- **Training:** the laptop adjusts model weights using examples and labels.
- **Inference:** the ESP32 uses those fixed weights to calculate a decision.
- **ANN:** an ordinary neural network using continuous hidden activations.
- **LIF neuron:** a software accumulator. Input raises its internal voltage,
  old voltage leaks away, and crossing a threshold emits a spike and resets it.
- **ALIF neuron:** a LIF neuron whose threshold temporarily rises after firing.
- **SNN:** connected spiking neurons plus a readout. Our model has sixteen
  hidden neurons, not just a single threshold rule.
- **Quantization:** storing weights with fewer bits, then checking whether the
  reduced precision changes predictions. Our states/arithmetic stay floating point.

## What happens from start to finish

1. Download verified public robotic-gripper recordings: three objects, ninety trials.
2. Convert changes in three tactile axes into six positive/negative channels.
3. Train on two objects, keeping separate trials for validation.
4. Test on the third object that the model has never seen. Repeat for all objects.
5. Compare logistic, ANN, frozen LIF, trained LIF and adaptive LIF models.
6. Check quantized predictions and export predetermined models into Arduino.
7. Upload using USB, send s, and save the ESP32 timing CSV with its metadata.

The raw sensor recordings are real. Our classification results were produced on
our computer. Synthetic stress tests are artificial. Hardware timing becomes a
result only after the actual board runs; a successful compile is preparation.

## A short viva explanation

“Our project studies spiking inference on a low-cost ESP32 for robotic-gripper
slip detection. We train on public tactile recordings and hold out complete
objects. We test whether temporal order matters, compare against an ordinary
network with memory, and check different window lengths and training rates.
Validation-selected LIF reached 62.9 percent balanced accuracy, versus 59.9 for
leaky-ReLU and 57.2 for the mean-activation ANN. Shuffling history reduced LIF
quality, but some training settings removed its advantage. The detector still
missed 82 percent of slip windows on the held-out solder spool. Our USB-only
Arduino demo runs the earlier frozen 32-sample model; these follow-up results are
computer-side. Physical onset, sensor transfer and energy remain unmeasured.”

## Likely questions

**Where is the novelty?**

SNN slip detection already exists. Our contribution is the controlled small-model
comparison and reproducible ESP32 deployment study. Do not claim a new neuron
or first-ever SNN slip detector. Publishable novelty would require broader evidence.

**Why use real gripper data if there is no gripper on the table?**

The application and recorded signals are from robotic gripping. The hardware demo
isolates computation on an ESP32. It does not reproduce the full physical system.

**Does 62.9% mean the detector is ready?**

No. It is modest balanced accuracy on this particular transfer task. Brush is
much easier than the other objects. The selected LIF misses about 82% of spool
slip windows despite its better aggregate score. Balanced accuracy is the average of slip
recall and non-slip recall, not the fraction of all windows classified correctly.

**How do you avoid leakage?**

Split entire objects/trials before fitting normalization or choosing thresholds.
Test windows never guide stopping or threshold selection. Since windows overlap,
resample whole trials for uncertainty, not individual windows.

**Why do we think timing matters?**

Keeping the final input and all input values unchanged, shuffling past frames
reduced frozen LIF quality. We also retrained on shuffled histories and observed
lower performance. The mean-activation ANN stayed unchanged. This supports
useful order information; it does not isolate every effect of spikes and reset.

**Are we choosing whichever test result looks best?**

No. Windows and training rates were selected using validation trials before test
inference. All candidates are reported, including worse settings. This remains
exploratory because the follow-up design was informed by our earlier study.

**Did adaptation work?**

Not better than LIF in this configuration. That is a research finding, not proof
that adaptive neurons never help. Our network is small and feed-forward.

**Why can a positive demo fixture produce a non-slip decision?**

It is the first predetermined positive window, selected without screening its
prediction. The models can fail. It checks runtime numerical agreement and timing;
accuracy is evaluated on all held-out windows rather than this repeated example.

**Is the SNN event-driven and energy-efficient?**

Its hidden neurons emit discrete spikes, but its input currents and input
accumulation are dense. The ESP32 simulates the neurons in software. We have no
measured energy result, and spike counts alone do not establish energy savings.

**Is int8 inference faster?**

We have established smaller logical parameter storage, not faster inference.
Weights are decoded into floating-point arithmetic and neuron states stay float32.
The board comparison will measure its actual execution cost.

**Can it detect slip within 5 ms?**

We cannot claim that. A 32-sample window requires roughly 178 ms of observation
at the source nominal rate. Fast computation does not remove that waiting time,
and the labels do not independently locate incipient onset.

**What is still needed?**

Identify the actual ESP32, upload, capture complete output and validate it.
Physical onset, continuous sensing, sensor transfer and gripping control are
separate future studies. The current computation demo needs only ESP32 and USB.
