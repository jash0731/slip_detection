# Experiment protocol, version 0.1

**Supporting acquisition protocol.** The clarified main project is neuromorphic
inference for robotic-gripper slip detection; see `RESEARCH_PROTOCOL.md`.
This file retains the original electrical GPIO experiment and its CSV schema.
SNN replay uses a separate schema documented in `SNN_HARDWARE.md`.

## Fixed definitions

- **Input pulse:** one generated digital active interval. The SW-420's actual active polarity must be confirmed on the physical module.
- **Target:** a burst of input pulses, with the pulse schedule stored in the trial log.
- **Noise:** isolated pulses or short patterns that the detector should ignore in the filtering experiment.
- **Detection time:** the first output decision for a trial.
- **Decision latency:** detection timestamp minus target burst's first pulse timestamp. Mark a trial with no decision by its deadline as `miss`; do not drop it from the dataset.
- **False alarm:** a decision during a noise-only trial or idle period.
- **Processing timestamp:** ESP32 timestamp taken at the detection decision, before serial output or LED animation.
- **Electrical loopback latency:** time between the timestamp at the GPIO generation call and the processing timestamp. This includes software instrumentation and is not mechanical slip latency.

## Experiments

### A. Acquisition method

Use the same immediate decision rule. Compare GPIO polling at 1, 5, 10, and 20 ms with interrupt acquisition. Generate isolated digital pulses of several widths, at varied phases relative to the polling schedule. At least 100 target trials per configuration. Report miss rate and latency median, 95th percentile, and maximum among detected trials. Record trial count alongside those statistics.

The first implementation is [`firmware/esp32_acquisition/esp32_acquisition.ino`](firmware/esp32_acquisition/esp32_acquisition.ino).
See [HARDWARE.md](HARDWARE.md) for wiring, upload, exact schedules, timestamp
boundaries, and CSV capture/analysis. It runs only Experiment A on a classic
dual-core ESP32; no hardware findings are supplied in the repository.

### B. Detection rule

Keep interrupt acquisition fixed. Feed identical pulse schedules to (1) immediate trigger, (2) N events in a time window, and (3) time-decaying LIF. Include target bursts and noise-only patterns. Tune N, window, LIF tau, and threshold on development patterns, then freeze all values before final trials. Report detection rate, false-alarm rate, and decision latency. Do not call the algorithms equivalent if one waits for multiple pulses.

LIF update at each input event: `V = V * exp(-elapsed_ms / tau_ms) + 1`. Fire and reset when `V >= threshold`. Initialise `V = 0`; elapsed time starts at the previous event. This is an event-domain temporal accumulator, not ADC sampling.

### C. Core placement

Keep input and detector settings fixed. Run the detector and a CPU workload on the same core, then on separate cores. Keep clock, workload, pulse schedule, Wi-Fi/Bluetooth state, logging, and task priorities fixed. Compare the latency distribution and missed events. Record actual core IDs. Core pinning is task placement, not a guarantee of timing isolation.

## Data columns for hardware run

`run_id,trial_id,trial_kind,acquisition,detector,poll_interval_ms,core_config,load,parameters,pulse_schedule_us,first_pulse_us,decision_us,latency_us,outcome,notes`

Possible `outcome` values: `detected`, `miss`, `false_alarm`, `correct_reject`. Preserve raw timestamps. Store a code version or commit identifier and wiring version in `run_id` or a separate metadata file.

## Four-person split

| Person | Primary deliverable | Review partner |
| --- | --- | --- |
| 1 | Board setup, wiring, sensor demo, photos | 2 |
| 2 | GPIO polling/interrupt acquisition, timestamps | 1 |
| 3 | Immediate/counter/LIF detectors, stimulus schedules | 4 |
| 4 | CSV analysis, plots, report and slides | 3 |

Everyone should be able to explain the measurement boundary and one limitation in the final review.

## First meeting checklist

- Agree on this working title and scope.
- Put the original PPT in a `references/` folder, clearly marked as an initial proposal rather than final evidence.
- Decide one repository owner and give all four teammates access.
- Assign the table above and a date for a 10-minute code/design review.
- Have each teammate run the simulation and explain one row of its output.
