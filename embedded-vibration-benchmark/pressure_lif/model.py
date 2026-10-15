from collections import deque
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Config:
    sample_ms: float = 5.
    acceleration_lowpass_ms: float = 100.
    pressure_lowpass_ms: float = 20.
    pressure_difference_ticks: int = 5
    acceleration_threshold_g: float = .025
    pressure_change_threshold: float = .015
    unloaded_adc: float = 100.
    contact_reference_adc: float = 2000.
    contact_on: float = .12
    contact_off: float = .08
    axis_tau_ms: float = 25.
    pressure_tau_ms: float = 30.
    output_tau_ms: float = 50.
    input_weight: float = 1.1
    acceleration_output_weight: float = .4
    pressure_increase_weight: float = .1
    pressure_decrease_weight: float = .1
    output_threshold: float = 1.
    axis_refractory_ticks: int = 2
    pressure_refractory_ticks: int = 5
    output_refractory_ticks: int = 10
    budget_ticks: int = 100
    budget_fraction: float = .25
    recent_acceleration_ticks: int = 20
    acceleration_lsb_g: float = .0039

    def __post_init__(self):
        if self.contact_reference_adc <= self.unloaded_adc:
            raise ValueError('Contact reference must exceed unloaded ADC')
        if not 0 <= self.contact_off < self.contact_on:
            raise ValueError('Contact gate requires ordered nonnegative thresholds')
        if self.budget_ticks != 100 or self.sample_ms != 5. or self.recent_acceleration_ticks != 20:
            raise ValueError('Initial protocol fixes 200 Hz, 500 ms budget and 100 ms acceleration guard')
        if self.budget_fraction != .25 or self.output_threshold <= 0:
            raise ValueError('Initial protocol fixes the pressure budget at 25% of a positive threshold')
        for value in [self.axis_tau_ms, self.pressure_tau_ms, self.output_tau_ms,
                      self.acceleration_lowpass_ms, self.pressure_lowpass_ms]:
            if value <= 0:
                raise ValueError('Time constants must be positive')
        for value in [self.pressure_increase_weight, self.pressure_decrease_weight]:
            if value < 0:
                raise ValueError('Pressure weights are magnitudes')


class LIF:
    def __init__(self, beta, refractory_ticks, threshold=1.):
        self.beta, self.refractory_ticks, self.threshold = beta, refractory_ticks, threshold
        self.reset()

    def reset(self):
        self.voltage, self.remaining = 0., 0

    def step(self, drive):
        if self.remaining:
            self.remaining -= 1
            self.voltage = 0.
            return False
        self.voltage = self.beta * self.voltage + drive
        if self.voltage >= self.threshold:
            self.reset()
            self.remaining = self.refractory_ticks - 1
            return True
        return False


class Kernel:
    def __init__(self, config=Config(), pressure_enabled=True):
        self.config, self.pressure_enabled = config, pressure_enabled
        self.axes = [LIF(math.exp(-config.sample_ms / config.axis_tau_ms),
                         config.axis_refractory_ticks) for _ in range(3)]
        self.pressure = [LIF(math.exp(-config.sample_ms / config.pressure_tau_ms),
                             config.pressure_refractory_ticks) for _ in range(2)]
        self.output = LIF(math.exp(-config.sample_ms / config.output_tau_ms),
                          config.output_refractory_ticks, config.output_threshold)
        self.reset()

    def reset(self):
        for neuron in [*self.axes, *self.pressure, self.output]:
            neuron.reset()
        self.tick = 0
        self.last_acceleration = None
        self.inhibition_history = deque([0.] * self.config.budget_ticks)
        self.excitation_history = deque([0.] * self.config.budget_ticks)

    def step(self, acceleration_events, increase_event, decrease_event, contact):
        if len(acceleration_events) != 3:
            raise ValueError('Three acceleration channels required')
        if not contact:
            self.reset()
            return dict(spike=False, acceleration_fired=False, inhibition=0., excitation=0.,
                        inhibition_500ms=0., excitation_500ms=0.)
        c = self.config
        self.inhibition_history.popleft()
        self.excitation_history.popleft()
        axis = [n.step(c.input_weight if e else 0.) for n, e in zip(self.axes, acceleration_events)]
        up = self.pressure_enabled and self.pressure[0].step(c.input_weight if increase_event else 0.)
        down = self.pressure_enabled and self.pressure[1].step(c.input_weight if decrease_event else 0.)
        recent = (self.last_acceleration is not None
                  and 0 < self.tick - self.last_acceleration <= c.recent_acceleration_ticks)
        cap = c.budget_fraction * c.output_threshold
        inhibition = (min(c.pressure_increase_weight, max(0., cap - sum(self.inhibition_history)))
                      if self.pressure_enabled and up else 0.)
        excitation = (min(c.pressure_decrease_weight, max(0., cap - sum(self.excitation_history)))
                      if self.pressure_enabled and down and recent else 0.)
        self.inhibition_history.append(inhibition)
        self.excitation_history.append(excitation)
        spike = self.output.step(c.acceleration_output_weight * sum(axis) + excitation - inhibition)
        if any(axis):
            self.last_acceleration = self.tick
        self.tick += 1
        return dict(spike=spike, acceleration_fired=any(axis), inhibition=inhibition,
                    excitation=excitation, inhibition_500ms=sum(self.inhibition_history),
                    excitation_500ms=sum(self.excitation_history))


class Preprocessor:
    def __init__(self, config=Config()):
        self.config = config
        self.acceleration_baseline = None
        self.pressure = None
        self.history = None
        self.contact = False

    def step(self, acceleration_g, fsr_adc):
        if len(acceleration_g) != 3 or not all(math.isfinite(v) for v in [*acceleration_g, fsr_adc]):
            raise ValueError('Finite three-axis acceleration and FSR values required')
        if not 0 <= fsr_adc <= 4095:
            raise ValueError('Expected 12-bit FSR ADC counts')
        c = self.config
        pressure = (fsr_adc - c.unloaded_adc) / (c.contact_reference_adc - c.unloaded_adc)
        if self.acceleration_baseline is None:
            self.acceleration_baseline = list(acceleration_g)
            self.pressure = pressure
            self.history = deque([pressure] * c.pressure_difference_ticks)
        a_beta = math.exp(-c.sample_ms / c.acceleration_lowpass_ms)
        self.acceleration_baseline = [a_beta * old + (1 - a_beta) * value
                                      for old, value in zip(self.acceleration_baseline, acceleration_g)]
        residual = [abs(value - baseline) for value, baseline
                    in zip(acceleration_g, self.acceleration_baseline)]
        p_beta = math.exp(-c.sample_ms / c.pressure_lowpass_ms)
        self.pressure = p_beta * self.pressure + (1 - p_beta) * pressure
        previous = self.history.popleft()
        self.history.append(self.pressure)
        change = self.pressure - previous
        self.contact = self.pressure >= (c.contact_off if self.contact else c.contact_on)
        return ([value >= c.acceleration_threshold_g for value in residual],
                change >= c.pressure_change_threshold,
                change <= -c.pressure_change_threshold, self.contact)


def merge_events(spike_times_ms, gap_ms=250.):
    events, last = [], None
    for time in spike_times_ms:
        if last is None or time - last > gap_ms:
            events.append(time)
        last = time
    return events


def score_positive(events_ms, onset_ms, end_ms, fps, scoring_start_ms=2000., scoring_stop_ms=6000.):
    if fps <= 0 or onset_ms is None or end_ms < onset_ms:
        raise ValueError('Independent onset/end annotations and positive FPS required')
    frame_ms = 1000. / fps
    events = [t for t in events_ms if scoring_start_ms <= t < scoring_stop_ms]
    deadline = min(onset_ms + 500., end_ms, scoring_stop_ms)
    matched = [t for t in events if onset_ms + frame_ms < t <= deadline]
    ambiguous = [t for t in events if abs(t - onset_ms) <= frame_ms and t <= deadline]
    return dict(conservative_detected=bool(matched), upper_bound_detected=bool(matched or ambiguous),
                ambiguous=bool(ambiguous), early_events=sum(t < onset_ms for t in events),
                delay_ms=None if not matched else matched[0] - onset_ms)


def config_dict(config=Config()):
    return asdict(config)
