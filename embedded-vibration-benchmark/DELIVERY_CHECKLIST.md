# Delivery checklist: robotic-gripper slip research

Target review: Friday evening, 9 October 2026. Updated as work is verified.
This checklist tracks deliverables, not a claim that physical validation is 80%
complete. The exact board is now identified. The user confirmed that a physical
gripper is not required; recorded-input ESP32 inference is the hardware demo.

## Research scope and evidence

- [x] Keep robotic-gripper incipient-slip detection as the application.
- [x] Explore prior work beyond a single LIF neuron.
- [x] Separate synthetic host results, measured ESP32 replay, and physical slip trials.
- [x] Inspect public tactile-slip data, actual labels, access, and licensing.
- [x] Select a defensible first dataset/task or explicitly limit the first study to controlled synthetic workloads.
- [x] Document hypotheses, comparisons, causal decision boundaries, and exclusions.

## Advanced extension completed on 7 October

- [x] Add targeted primary-source research on surrogate gradients, adaptive neurons and quantization.
- [x] Download/checksum all three objects: 90 trials, 228,521 raw samples.
- [x] Audit causal preprocessing and object/trial splits independently.
- [x] Train matched 6→16→1 ANN, LIF and adaptive LIF with fixed seeds/budgets.
- [x] Add logistic and same-initialization frozen-hidden LIF controls.
- [x] Evaluate three held-out objects and retain all raw predictions/checkpoints.
- [x] Estimate paired uncertainty with 2,000 whole-trial bootstrap replicates.
- [x] Audit int8 weight storage and held-out decision drift against float models.
- [x] Export float/int8-weight models and verify all six C++ reference combinations.
- [x] Compile advanced Arduino replay for classic ESP32; no warnings.
- [x] Add complete-capture validator and run all 40 Python checks.
- [x] Write current report, three inspected scientific plots, presentation outline and viva guide.
- [ ] Confirm actual board, upload advanced sketch, collect/validate complete CSV over several boots.
- [ ] Insert actual board timing summaries into report/slides, keeping host findings distinct.

The USB-only model replay is the prepared hardware demonstration. If a live
vibration/LED acquisition demonstration is also used, sensor identification and
its separate setup are still pending; no trained-model transfer is assumed.

## Reproducible software

- [x] Complete fixed first-three follow-up protocol before new evaluation.
- [x] Test original checkpoints with three past-frame shuffles and reversal.
- [x] Add a capacity-matched conventional leaky-ReLU temporal network.
- [x] Train 108 fixed window/rate/model candidates and 12 shuffled-history controls.
- [x] Compare 16/32/64-sample windows on identical endpoint labels and original trial splits.
- [x] Select configurations using validation only; save choices before test inference.
- [x] Calculate paired whole-trial uncertainty for selected models and temporal controls.
- [x] Audit all 60,090 original prediction rows for misses, false alarms and context failures.
- [x] Record selected-model error rates and original timestamp observation costs.
- [x] Save follow-up report, raw artifacts, four inspected figures and reproduction commands.
- [x] Run all 50 Python tests; verify original deployed export/predictions remain frozen.
- [x] Implement conventional controls and a complete small spiking network.
- [x] Document what is trained, what stays fixed, and recording-level splits.
- [x] Compare dense and sparse accumulation on identical weights and inputs.
- [x] Verify output/state equivalence, reset behavior, and deterministic export.
- [x] Save raw CSV, parameters, seeds, and machine/runtime metadata.
- [x] Produce analysis with limitations; never present synthetic accuracy as slip accuracy.

## ESP32 preparation

- [x] Existing GPIO polling/interrupt loopback sketch and raw capture analyzer.
- [x] Existing classic-ESP32 loopback sketch compile verified; no board run claimed.
- [x] Add portable Arduino SNN replay sketch, with static exported weights and state.
- [x] Verify host C++ against the offline reference.
- [x] Compile the replay sketch using the ESP32 board package.
- [x] Explain USB-only replay, capture fields, timing boundary, and sensor limitations.
- [x] Add and verify an on-board controlled activity sweep for H1 (code/compile; measurements pending).
- [x] Identify board as ESP32-D0WD-V3; upload and validate actual board capture.

## Report and review package

- [x] Update the project entry point to reflect robotic grippers and neuromorphic inference.
- [x] Write a research-report draft with verified software findings and explicit pending measurements.
- [x] Prepare presentation content and an explanation suitable for the course review.
- [x] Run the complete verification suite and record reproducible commands.

## Physical validation: optional future extension, outside Friday's scope

- [ ] Choose tactile sensor and conditioning based on the chosen signal/task.
- [ ] Build gripper/contact setup and independently label slip onset.
- [ ] Collect held-out grasping trials across objects/materials.
- [ ] Measure false alarms, misses, onset delay, and full sensor-to-decision latency.
- [ ] Evaluate grip adjustment only after detection is validated.
- [ ] Measure energy with suitable equipment if energy becomes a reported outcome.

No physical gripper, sensor acquisition, on-board SNN measurements, energy data,
or closed-loop grip control is supplied by the pre-hardware software package.

The first three software investigations are finished within the bounded
[follow-up protocol](FOLLOWUP_PROTOCOL.md); this does not close the broader
scientific question. Important remaining uncertainty includes poor object transfer,
training sensitivity, spatial-feature loss and limited independent source objects.
Actual upload/capture remains the next required hardware step. Follow-up selected
models have not replaced the compiled32-sample board demonstration.
