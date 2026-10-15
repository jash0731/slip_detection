# Pressure-conditioned LIF: executable prototype and final analysis

The uploaded ESP32 protocol has been implemented as a Python reference and a matching host C++ network kernel. Software validation passed. Real slip recall, false-alarm rate, balanced accuracy, ESP32 timing and the protocol's primary success target have not been measured.

This is a new sensor experiment using jaw acceleration plus an FSR proxy. Existing CT-LIF results from the ARQ tactile-array recordings cannot be reused as results for this method. The local hardware_results logs are older public-data replay experiments, not ADXL345/FSR recordings from the specified fixture.

See [the supplied protocol](protocol.md), [starting parameters](starting_config.json), and [machine-readable software checks](software_validation.json).

## What was tested

Eleven software checks passed, including:

- Both rolling pressure budgets stayed at or below 0.25 of the output threshold over randomized and repeated-spike stress inputs.
- Pressure decrease could not excite the network without an acceleration neuron firing in the preceding 100 ms. Acceleration on the same tick alone does not satisfy this guard.
- Acceleration alone reached the output threshold under constant pressure.
- Contact loss reset network state and recent-acceleration history.
- The zero-pressure-weight ablation exactly matched baseline output spikes on 20,000 randomized updates.
- Python and host C++ default kernels produced zero output-spike mismatches over 20,000 randomized updates; pressure-pulse values agreed within 1e-12.
- CSV replay accepted a complete synthetic hold and rejected a sequence gap.
- Event helpers distinguished early alarms from post-onset detections and handled camera-frame timing ambiguity.

These are software checks on constructed inputs, not 20,000 labelled slip observations, and not an accuracy percentage. C++ validation used host double arithmetic; it is not an ESP32 execution or float-precision validation.

## Diagnostic findings

All times below are generated from synthetic input-spike patterns. They are not measured physical detection delays.

| Controlled input | Baseline | Proposed | Implication |
|---|---|---|---|
| Pressure changes without acceleration | No output | No output | The recent-acceleration gate blocks pressure-only excitation |
| Constant pressure with a four-pulse acceleration burst | Output at 45 ms | Output at 45 ms | Maintained-pressure activity can reach threshold through acceleration alone |
| Same weak burst with pressure-increase pulses | Output at 45 ms | No output | A legal inhibition budget can still suppress weak evidence |
| Three subthreshold acceleration pulses, then pressure decrease | No output | Output at 20 ms | Pressure excitation can create an alarm after a non-slip disturbance if those sensor events occur |
| One simultaneous three-axis impact | Immediate output | Immediate output | The default network does not automatically reject taps |

The inhibition counterexample used only 0.20 of threshold, below the 0.25 limit. The excitation counterexample used 0.10 of threshold. Thus the caps limit intervention but do not prove preserved slip recall or low false alarms. These tests confirm the failure modes anticipated in the supplied protocol.

The likely benefit is suppression of some pressure-increase disturbances. The main risks are missing weak squeeze-plus-slide events and generating alarms during pressure relaxation after recent vibration. Whether the benefit outweighs the risks depends on independently labelled recordings.

## Exact implementation candidates

All parameters are starting candidates, not development-tuned or test-frozen values. The protocol specifies structure but leaves exact weights and time constants for development.

| Setting | Prototype value |
|---|---:|
| Sample interval | 5 ms, 200 Hz |
| Acceleration low-pass time constant | 100 ms |
| Pressure low-pass time constant | 20 ms |
| Pressure-difference lag | 5 ticks, 25 ms |
| Acceleration residual threshold | 0.025 g |
| Normalized pressure-change threshold | 0.015 |
| Axis / pressure / output LIF time constants | 25 / 30 / 50 ms |
| Input-spike weight into input neurons | 1.1 |
| Axis spike weight into output | 0.4 |
| Pressure increase / decrease output magnitude | 0.1 / 0.1 |
| Output threshold | 1.0 |
| Axis / pressure / output refractory periods | 10 / 25 / 50 ms |
| Contact gate on / off | 0.12 / 0.08 normalized proxy |
| Independent excitation / inhibition cap | 0.25 each per rolling 500 ms |
| Pressure excitation guard | Prior acceleration firing within 100 ms |

The exact ADC references, sensor conversion factor and remaining constants are in starting_config.json. ADC references of 100 and 2000 counts are placeholders that require actual development measurements. The acceleration conversion factor must match the configured ADXL345 mode and range. No force calibration in Newtons is claimed.

Each neuron decays its voltage exponentially at each tick, adds weighted input spikes, emits at threshold and resets to zero. During refractory ticks it holds zero and ignores drive. Baseline uses three acceleration neurons and one output neuron. Proposed adds pressure-increase and pressure-decrease neurons. The baseline skips those two updates. The generic reference object still reserves their fields; a dedicated embedded baseline can omit unused state. No memory superiority is claimed from these generic objects.

Excitation and inhibition use independent exact 100-tick histories. This simple prototype sums the histories each tick; it is not optimized for hardware speed. The C++ header implements the default constants only. If development changes them, regenerate or edit the header and rerun parity tests before using it on hardware.

## Run software validation and replay

From embedded-vibration-benchmark, with Python, NumPy and g++ available:

```bash
python -B -m pressure_lif.validate
python -B -m pressure_lif.replay recordings.csv --config pressure_lif/starting_config.json
```

Validation compiles the host kernel into a temporary directory and removes the binary afterward. It prints JSON and writes no file by default. To save a new software report:

```bash
python -B -m pressure_lif.validate --output new_software_validation.json
```

Required recording columns are trial_id, seq, t_us, ax_raw, ay_raw, az_raw, fsr_adc. The replay checks consecutive sequence numbers, increasing timestamps and an explicitly declared 4-6 ms spacing range around the nominal 5 ms update. Signals are not silently resampled. Change and disclose the jitter policy only during development if real acquisition requires it. Replay output timestamps retain the recording's t_us time base, converted to milliseconds; first_t_us is returned for alignment. Independent video labels and scoring bounds must use the same time base.

The replay emits baseline/proposed spike and merged-warning timestamps. It does not infer ground-truth labels from sensors, perform a 30-candidate search, automatically annotate video, or produce final trial-performance tables. model.score_positive is a scoring helper: callers must supply independently annotated onset/end times, actual video FPS and appropriate scoring bounds. Its conservative match excludes a full frame on either side of onset; ambiguous matches contribute only to the upper bound. LED hold duration never substitutes for spike onset.

The host C++ header is an inference kernel, not a complete ESP32 sketch. ADXL345 register setup, fresh-reading checks, ADC acquisition, synchronization LED sequence, logging, live event timestamps and embedded compilation remain to be integrated and physically tested.

## What is needed for a measured final result

The supplied protocol requires a 15-trial pilot, followed by 42 development trials and 84 held-out test trials. The test set contains 36 positive and 48 negative trials, including 24 squeeze/regrip or loosen-without-slide negative trials used for the primary comparison. The 15 pilot trials and procedural failures are retained separately.

Tune each model on development data with the same cap of 30 candidates, then freeze both before collecting the test session. The development set has 18 positives: reaching at least 85% recall requires at least 16/18 detections. No such tuning or freezing was performed here because those recordings are absent.

Primary success requires at least 30% relative reduction in false-alarm trials among the 24 pressure-disturbance negatives, while losing at most one detection among 36 positive trials. Losing two gives a 5.56-percentage-point recall decrease and fails the five-point guard. If baseline has no primary false alarms, relative reduction is undefined. This is the protocol's target, not a measured result.

Report maintained-grip slides and squeeze-plus-slide separately. Keep timing-ambiguous trials in denominators and report conservative and upper-bound recall. Compare proposed both to an independently tuned baseline and to its own zero-pressure ablation. Use new test recordings after any revision based on development findings; do not retune on final test outcomes.

| Required empirical result | Current status |
|---|---|
| ADXL345/FSR detectability in this fixture | Not tested |
| Slip recall, false alarms and balanced accuracy | Not measured |
| Primary 30% reduction target | Not evaluated |
| Software/ESP32 event agreement | Not tested; host C++/Python agreement only |
| ESP32 mean/max inference time and missed deadlines | Not measured |
| Standalone warnings and three power-cycle checks | Not tested |
| Publication novelty | Not established |

## Final assessment

The design is executable and its proposed safeguards can be implemented consistently. It is suitable to take into the pilot, with the stated pressure-related failure modes treated as mandatory tests. It is not yet supported as an accuracy improvement over LIF, CT-LIF or the nonlinear model, and it is not ready to report measured ESP32 slip-detection performance. The honest present result is a validated software prototype with explicit counterexamples and an untested physical hypothesis.
