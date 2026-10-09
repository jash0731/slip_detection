# Follow-up findings: temporal mechanism, robustness and failure modes

Completed bounded software investigation, 7 October 2026. Robotic-gripper slip
recognition remains the application. All classifications use the authors' public
recordings on our computer; no ESP32 or physical sensing measurements are added.
The original report, predictions and Arduino export remain frozen.

## What the new investigation establishes

Past-frame order contributes to this trained LIF configuration's quality. A
matched conventional temporal network improves over the mean-activation ANN,
but LIF retains an advantage in the declared comparisons. That advantage is
sensitive to training rate, and good aggregate balanced accuracy hides serious
object-specific failures. These conclusions are more informative than adding
architectures alone; they still concern three objects and one sensor setup.

We fixed the [protocol](FOLLOWUP_PROTOCOL.md) before new evaluation. It includes
108 window/rate/model candidate fits, 12 shuffled-training controls, frozen
checkpoint perturbations and a post-hoc audit of every original prediction.
The six original object/seed trial partitions and training-only normalization
are reused. Candidate histories and validation choices are saved before test
inference for each fold. Nothing is selected by outer-test score.

## 1. Does temporal order matter?

The new conventional baseline has the same 6→16→1 initialization and 129 learned
parameters as ANN/LIF. Its state is
`h[t] = ReLU(0.875*h[t-1] + W*x[t] + b)` and the linear readout averages that
activity. Its diagonal recurrence is fixed; there is no learned recurrent matrix.
It supplies memory without spikes or reset, with the same training/validation rules.
This is a controlled small baseline, not an optimized TCN or full recurrent ANN.

For the original frozen checkpoints, we jointly permute the past 31 six-channel
frames, preserving the final frame, labels and input multiset. We average three
declared shuffle draws, retaining the original validation thresholds. Past reversal
is a separate diagnostic. These altered windows are offline ablations, not plausible
physical sensor recordings or deployment inputs.

| Frozen model | Original order | Reversed past | Shuffled past, three draws |
| --- | ---: | ---: | ---: |
| Mean-ReLU ANN | 57.21% | 57.21% | 57.21% |
| LIF | **60.78%** | 60.41% | 58.87% |
| ALIF | 57.51% | 57.39% | 57.25% |

ANN changes no decisions, as expected for a mean-activation representation.
Float roundoff reaches approximately 1.4e-6 in logits. LIF changes 1,491 of
36,054 shuffled evaluations, a 4.14% disagreement rate. Its mean balanced accuracy
drops **1.91 percentage points**, with a conditional paired whole-trial 95% interval
of **0.87–3.14 points**. Reversing past frames produces a much smaller 0.38-point
drop. ALIF's shuffled drop is only 0.26 points and its conditional interval includes
zero. Sensitivity depends on how history is perturbed, not just whether it changes.

Test-only perturbation could impose distribution shift. We therefore also train
and calibrate shuffled-history LIF/leaky-ReLU models, at the prespecified
window32/rate 0.01, before testing similarly shuffled windows:

| Model, matched fixed configuration | Ordered train/validation/test | Shuffled train/validation/test |
| --- | ---: | ---: |
| LIF | 60.78% | 57.25% |
| Leaky-ReLU | 59.04% | 57.64% |

Ordered LIF exceeds its shuffled-training control by **3.54 points** (conditional
95% interval 1.73–5.32); leaky-ReLU exceeds its control by 1.40 points (0.13–2.42).
At matched window32/rate 0.01, ordered LIF exceeds ordered leaky-ReLU by 1.75 points
(0.16–3.67). This supports usefulness of temporal structure under this encoding.
It does not isolate spikes from reset, gradient conventions, initialization or
every optimization effect, nor establish a biological mechanism.

![Order controls and matched retraining](followup/figures/temporal_controls.png)

## 2. Is the advantage robust?

The fixed grid uses windows of 16/32/64 samples and Adam rates 0.005/0.01 for ANN/LIF/leaky-ReLU,
two seeds and three held-out objects. Maximum 18 epochs, patience4 and all other
settings match the first study. The declared 120 total fits completed; training
wall time was about 77 seconds on the development computer, not inference timing.

Every partition shares endpoints end_sample ≥ 63 at stride 16 across all windows.
The actual dataset retains the same 6,009 eligible test endpoints. All longer
windows supply additional past context, not different test labels. Choose each
model's window/rate by validation balanced accuracy, with declared tie rules,
before test inference. Validation tends to favor longer windows; those carry
more observation cost. Test-grid maxima are not used to pick a winner.

| Validation-selected model | Mean object/seed balanced accuracy |
| --- | ---: |
| Mean-ReLU ANN | 57.22% |
| Leaky-ReLU temporal ANN | 59.94% |
| LIF | **62.86%** |

Selected LIF minus ANN is **+5.63 points**, conditional paired-trial 95% interval
**+3.44 to +7.66**. Selected LIF minus leaky-ReLU is **+2.92 points**, interval
**+1.33 to +4.57**. These comparisons allow different validation-selected windows;
the matched32-sample comparison above addresses equal-horizon quality separately.

| Held-out object | Selected ANN | Selected leaky-ReLU | Selected LIF |
| --- | ---: | ---: | ---: |
| Brush | 63.65% | 70.15% | 75.67% |
| Screwdriver | 54.32% | 54.09% | 55.95% |
| Solder spool | 53.70% | 55.58% | 56.96% |

The fixed-grid sensitivity result tempers that aggregate advantage. At rate 0.01,
LIF scores 60.78–62.19% across the three windows. At rate 0.005 it scores 57.66–58.98%,
and is slightly below ANN at windows of 16/32 samples. Thus **LIF does not win every declared
configuration**, and increasing context is not monotonically beneficial. Selected
quality is a property of this validation procedure, not an architecture guarantee.
No candidate threshold lies at the edge of the prescribed threshold grid.

![Window and training-rate sensitivity](followup/figures/robustness_grid.png)

The source timestamps also confirm a real observation trade-off. Median raw signal
support is about 88.9/177.8/355.6 ms for 16/32/64 samples across the objects, including
the preceding sample needed for first differences. Per-object p95 and extrema are
saved in `observation_summary.csv`. These are durations of recorded input support,
not physical slip lead time or ESP32 detection latency.

## 3. Where does the detector fail?

The audit retains all 60,090 original prediction rows: five models × two seeds ×
6,009 unique windows. Per-trial/object tables report confusion counts, slip recall,
non-slip recall, precision, miss and false-alarm rates. Single-class groups have
undefined balanced accuracy. Context strata retain sample counts and eligible
trial counts; they are descriptive post-hoc groups, not independent tests.

The selected-model error table makes the remaining limitations clear:

| Selected LIF, mean of two seeds | Slip recall | Miss rate on slip | False-alarm rate on non-slip |
| --- | ---: | ---: | ---: |
| Brush | 79.4% | 20.6% | 28.0% |
| Screwdriver | 57.8% | 42.2% | 45.9% |
| Solder spool | **18.0%** | **82.0%** | 4.1% |

The spool balanced-accuracy improvement largely reflects good non-slip specificity,
while recall remains poor. Reporting the higher overall score alone would hide an
unsuitable slip alarm. Even brush has substantial false alarms; these results
cannot support deployment-ready grip protection.

For the original frozen window32 LIF, false alarms are particularly high in mixed
held-state context: approximately 64.6% on brush and 70.1% on screwdriver. Sample
counts are small in some context strata, so those rates are descriptive. Stable
spool slip still has about 78.9% misses, showing that the failure is not confined to
annotation transitions. Label-run age tables use annotated history, not a ground-
truth incipient onset. Consecutive-error tables never bridge unscored endpoint gaps.
Illustrative errors are chosen by largest wrong logit margin and explicitly marked
as examples, not representative or calibrated confidence estimates.

![Selected quality and misses](followup/figures/selected_quality_and_misses.png)

![Original frozen LIF context errors](followup/figures/failure_context.png)

Feature-shift analysis gives a possible explanation to investigate, not a causal
proof. Relative to training p95 derivative scales, held-out screwdriver p95 is
about 1.72/2.31/1.40 on x/y/z; spool is about 0.54/0.39/0.54. Weaker normalized spool
signals and stronger screwdriver signals are consistent with misses and false
alarms, respectively. This analysis includes all raw context rows and compares
different objects, so it cannot disentangle material/contact behaviour, grasp pose,
annotation and preprocessing. Averaging eighteen pins also loses spatial structure.

## Strength and limits of the evidence

Paired intervals use 2,000 whole-grasp-trial resamples within each object, shared
across methods and fixed seeds. All requested draws were retained. Shuffled
diagnostics average their three declared draws; those are not independent trials.
Intervals condition on these observed objects and fitted models, excluding
retraining/selection uncertainty and arbitrary new-object variation. Two seeds
also change inner validation splits. Comparisons remain exploratory, without
multiple-testing correction or a universal superiority claim.

The first three software investigations are complete within their declared bounds.
We now have evidence that timing information helps, that learning settings matter,
and that aggregate accuracy conceals severe transfer failures. Further scientific
work should test spatial/contact features and additional independent objects with
validation-only selection, rather than simply adding more neurons.

The existing Arduino export remains the original frozen window32 demonstration.
These selected 16/32/64 follow-up checkpoints have not been exported or compiled for
the board. Actual ESP32 timings, live vibration-sensor transfer, physical incipient
onset, energy and grip control remain unmeasured. See [reproduction instructions](FOLLOWUP_SOFTWARE.md)
and the [verification record](VERIFICATION.md).
