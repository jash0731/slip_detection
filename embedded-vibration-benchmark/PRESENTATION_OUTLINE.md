# Course review presentation content

Updated 7 October 2026. Suggested thirteen slides, including the follow-up. This is editable slide content,
not a finished PowerPoint. Keep source/limit captions visible on plots.

## 1. Project and question

**Trained spiking inference on ESP32 for robotic-gripper slip detection**

How do training, adaptive neurons and reduced weight precision affect quality
and deployment cost on a low-cost microcontroller? Gripper sensing remains the
application; recorded-input inference is the course hardware demonstration.

## 2. Why this is research

Spiking slip detection already exists. Surrogate gradients enable hidden-weight
training; adaptive thresholds may improve temporal behavior; int8 storage may
save bytes. On an ordinary ESP32, none automatically guarantees accuracy, speed
or energy savings. We test controlled hypotheses and retain unfavorable results.
Use the primary-source links in RESEARCH_UPGRADE.md in slide notes.

## 3. Data and honest task definition

ARQ-CRISP robotic-gripper tactile data: three objects, ninety original trials,
228,521 raw samples, 54 channels. Source labels distinguish held non-slip/slip.
They do not independently locate physical incipient onset. Cite Zenha et al.,
IROS 2021 and the source repository. Cheap vibration-sensor transfer is untested.

## 4. Leakage prevention and transfer

Train on two objects; test on the third, repeated for all three objects.
Within training objects, hold out whole trials for validation. Normalize only
using training data and select epochs/thresholds only on validation.
Causal 32-sample windows, stride16; earlier transitions remain context.
6,009 unique test windows across 87 eligible trials. Windows overlap; bootstrap
entire trials instead of treating windows as independent samples.

## 5. Matched models and ablations

Six continuous signed-difference channels → sixteen hidden units → one readout.
ANN/LIF/ALIF each learn 129 parameters. LIF/ALIF have hard spikes in inference
and surrogate gradients during training. Frozen LIF isolates hidden learning;
logistic regression provides a simple trained control. State resets each window.
Two fixed seeds; bounded eighteen-epoch training with validation early stopping.

## 6. Actual classification finding

Insert advanced/figures/cross_object_quality.png.

Object/seed mean balanced accuracy: logistic57.3%, ANN57.2%, frozenLIF57.4%,
trainedLIF60.8%, ALIF57.5%. LIF is strongest here; transfer is much weaker on
screwdriver and solder spool than brush. Adaptation does not improve the aggregate.
These are public recorded host results, not measurements from our ESP32.

## 7. Uncertainty and interpretation

Insert advanced/figures/paired_trial_intervals.png.

Two thousand paired whole-trial resamples. LIF−ANN:+3.58 percentage points,
conditional95% interval +1.51 to +5.70. LIF−frozen:+3.38; ALIF−LIF:−3.28.
Intervals condition on the three objects and fixed models; they exclude training
and arbitrary new-object uncertainty. No universal architecture claim.

## 8. Quantization and the trade-off

Insert advanced/figures/weight_quantization.png.

Logical parameter storage516→188bytes, about63.6% reduction. Int8 weights,
float32 biases/states/arithmetic. Across12,018 evaluations per model, decisions
changed:LIF48, ALIF15, ANN11. Accuracy can move either way. Smaller parameter
storage does not prove smaller whole firmware, faster execution or lower energy.

## 9. ESP32 demonstration and evidence

Upload firmware/esp32_advanced_replay, Serial115200, send s. USB only.
Three models × two precisions × ten repetitions ×32steps =1,920 CSV rows.
Preflight and measured traces check numerical parity against Python references.
Compile passed on classic ESP32:278,136flashbytes/23,252staticRAMbytes.
Actual board timing remains pending. After upload show the authentic Serial
capture and validated summaries, including board/toolchain identifiers.

The fixed positive fixture is misclassified by all three exported models; it
is retained without screening. It demonstrates compute/parity, not accuracy.
Timing excludes raw tactile preprocessing and physical sensing.

## 10. What we learned and what remains

In this limited study, hidden-trained LIF improves the matched comparison,
adaptation does not, and smaller weights require a numerical-quality audit.
The observation window is roughly178ms at the nominal source rate; compute time
is not physical onset latency. Upload/capture is the next required course check.
Incipient onset, continuous operation, sensor transfer, energy and grip control
remain future work. No motor or physical gripper is needed for the current demo.

## Viva preparation

Use VIVA_GUIDE.md for a short explanation and likely questions. Distinguish
balanced accuracy from ordinary accuracy. Never claim guaranteed energy savings,
5ms slip prediction or working grip control. Explain limitations directly: a
negative result can still answer a well-designed research question.

## Follow-up slides to insert after the initial comparison

### Temporal mechanism

Insert followup/figures/temporal_controls.png.

Preserve final frame and input values; shuffle/reverse past frames jointly.
Frozen LIF 60.78→58.87% under shuffling; ANN unchanged. With matched shuffled
training, LIF falls to 57.25% and leaky-ReLU to 57.64%. These are order ablations,
not physical sensor recordings or proof of a biological mechanism.

### Robustness and fair selection

Insert followup/figures/robustness_grid.png.

108 candidates across 16/32/64-sample windows, two learning rates, three models,
three held-out objects and two seeds; plus 12 shuffled-training fits. Identical
endpoint labels. Validation-selected means: LIF 62.86%, leaky-ReLU 59.94%, ANN 57.22%.
LIF does not win every training setting. Selected models can use different
windows, so observation cost differs. No winner is selected by test score.

### Failure modes are a finding

Insert followup/figures/selected_quality_and_misses.png.

Selected LIF still misses 82% of held-out spool slip windows and falsely alarms
on about46% of screwdriver non-slip windows. Aggregate balanced accuracy alone
hides these failures. Feature shifts are a possible explanation to investigate,
not a causal proof. All follow-up results are computer-side; the Arduino export
remains the original frozen 32-sample demonstration. Fifty Python tests pass.

The original slides retain fixed 32 results and quantization provenance; identify
that experiment explicitly instead of mixing it with validation-selected scores.
Use FOLLOWUP_REPORT.md for the updated scientific conclusion and limitations.
