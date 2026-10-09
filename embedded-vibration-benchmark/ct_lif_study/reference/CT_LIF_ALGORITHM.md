# CT-LIF: Cancellation–Turnover Encoding for Compact Tactile Slip Recognition

**Candidate algorithm specification, version 0.1 — 8 October 2026**

**Evidence status:** designed and implemented; synthetic implementation tests passed. Not trained or evaluated on the project's real recordings in this work. Novelty remains unresolved. No claim that this is unpublished worldwide, better than existing models, or ready for ESP32 deployment.

## 1. The idea in easy English

Your current pipeline averages 18 tactile pins before calculating changes. Imagine two pins change by +5 and −5. Their average is zero, even though both moved. Once this detail is averaged away, a neural network cannot recover it.

CT-LIF asks three questions before throwing away pin information:

1. **Common motion:** are the pins mostly changing in the same direction?
2. **Cancellation:** how much activity disappears when positive and negative pin changes are averaged?
3. **Turnover:** are different pins becoming active from one moment to the next, while cancellation is present?

It converts those measurements into 12 small input channels and feeds them to an ordinary 16-neuron LIF network. A learned output produces a slip score. The important candidate contribution is the encoder; LIF itself is established prior work.

Two pins moving differently does not prove slip. Grip adjustment, vibration, sensor noise and normal contact deformation may also produce these measurements. Training and controlled comparisons must establish whether they are useful.

## 2. Why this fits your reported failure

Your `FOLLOWUP_REPORT.md` reports selected-LIF balanced accuracy of 62.86%, solder-spool slip recall of 18%, and signal-scale differences across held-out objects. It explicitly notes that averaging pins loses spatial information. These motivate two hypotheses:

- A normalized description of *how* pins move may transfer better than a description dominated by absolute signal strength.
- Pin activity hidden by averaging may contain useful classification information.

Neither hypothesis has been established by the existing report. The new method may fail, particularly if the apparent lost activity is mostly noise.

This is an extension for the recorded **18-pin, three-axis** dataset. A single piezo or SW-420 sensor cannot provide its required pin patterns. Such hardware would require a different encoder and a separate validation study.

## 3. Exact input and notation

At sample t, x[p,a,t] is the raw value for pin p=1,…,18 and axis a∈{x,y,z}. Pin identities must be consistent over time. All operations use current/past samples only. No pin coordinates, contact geometry, object identity or ground-truth slip labels enter the inference calculation.

Define the per-pin first difference:

\[
d_{p,a,t}=x_{p,a,t}-x_{p,a,t-1}.
\]

Define signed mean motion and mean absolute activity:

\[
m_{a,t}=\frac1{18}\sum_p d_{p,a,t},\qquad
A_{a,t}=\frac1{18}\sum_p|d_{p,a,t}|.
\]

The scale s[a] is the training-only 95th percentile of A[a,t]. In this implementation it is fitted over the supplied training windows, including their derivative context, so overlapping samples receive repeated weight. This is deliberate and must be held fixed across candidate comparisons; fitting once on unique training rows would be a separate preprocessing revision. If any axis has zero p95 variation, fitting fails explicitly.

Set η=0.02 initially. It is a design choice, **not a measured sensor noise floor**. A frame is numerically active when A>10⁻¹²s. Otherwise its normalized features are zero.

## 4. The three measurements

### 4.1 Common motion

For active frames:

\[
q_{a,t}=m_{a,t}/A_{a,t},\qquad
r_{a,t}=\min(1,A_{a,t}/(\eta s_a)).
\]

q ranges from −1 to +1. The gate r prevents arbitrarily tiny activity from producing a full-strength feature. It does not independently determine sensor reliability or distinguish noise from slip.

Retain the two sign channels:

\[
f^+_{a,t}=r_{a,t}\max(q_{a,t},0),\qquad
f^-_{a,t}=r_{a,t}\max(-q_{a,t},0).
\]

These six channels retain the direction of average motion relative to total activity. They deliberately discard most absolute amplitude above the gate.

### 4.2 Cancellation

\[
C_{a,t}=1-|q_{a,t}|,\qquad
f^C_{a,t}=r_{a,t}C_{a,t}.
\]

C is zero when all nonzero changes have the same sign, and one when positive and negative changes exactly cancel. It is zero by convention for inactive frames.

This formula is a standard normalized cancellation/polarity-balance quantity, **not a newly invented mathematical primitive**. If P and N are total positive and negative magnitudes, C=2min(P,N)/(P+N).

### 4.3 Turnover of pin activity and its coupling to cancellation

For active frames define the distribution of absolute activity across pin identities:

\[
\pi_{p,a,t}=\frac{|d_{p,a,t}|}{\sum_j|d_{j,a,t}|}.
\]

Then compute total variation distance between consecutive distributions:

\[
V_{a,t}=\frac12\sum_p|\pi_{p,a,t}-\pi_{p,a,t-1}|.
\]

V=0 if activity stays distributed identically; V=1 if it moves to disjoint sets of pins. Set V=0 if either frame is inactive. This distance is standard mathematics.

The proposed coupled feature is:

\[
\boxed{f^{CT}_{a,t}=\min(r_{a,t},r_{a,t-1})\,C_{a,t}\,V_{a,t}}.
\]

It emphasizes redistribution of activity specifically when averaging is hiding opposing motion. This coupling, together with the other nine channels and compact temporal classifier, is the narrow candidate to investigate. There is no established physical law requiring slip to produce a high value.

The final channel order is:

\[
[f_x^+,f_y^+,f_z^+,f_x^-,f_y^-,f_z^-,f_x^C,f_y^C,f_z^C,f_x^{CT},f_y^{CT},f_z^{CT}].
\]

All channels lie in [0,1]. Zero-motion frames map to zero.

## 5. A worked example

Use four pins here only to make the arithmetic visible. The implementation uses 18.

| Axis changes across pins | Signed mean m | Absolute mean A | Cancellation C |
|---|---:|---:|---:|
| +1, +1, +1, +1 | 1 | 1 | 0 |
| +1, −1, 0, 0 | 0 | 0.5 | 1 |
| 0, 0, +1, −1 | 0 | 0.5 | 1 |

In the last two rows the old mean representation is zero. Their activity distributions are [0.5,0.5,0,0] and [0,0,0.5,0.5], so V=1. With saturated gates, fCT=1.

Both frames have the same multiset of values and the same activity entropy. A change in entropy would be zero, but turnover is one because it retains correspondence between pin identities across time. This is a representational distinction; it does not establish novelty or predictive value.

## 6. What is invariant, and what is not

If the derivative values on an axis are multiplied by a positive common gain g, m and A scale by g, while q, C, π and V do not change. The full encoder is therefore invariant to that gain **only while both relevant gates remain saturated** and the numerical activity test is unchanged. For raw readings, the same statement follows from a constant positive gain over the relevant samples; time-varying gains introduce extra derivative terms.

The invariance is per axis and assumes a common gain across pins. It does not cover arbitrary pin-specific gains, rotation of coordinate axes, additive noise, sensor saturation, missing pins, nonlinear material effects or different contact mechanics. Below the gate, outputs change with gain by design.

A fixed permutation of pin labels leaves the features unchanged, provided it is consistent across time. A changing permutation can produce false turnover. Constant per-pin offsets disappear through differentiation.

This encoder retains a measure of changing pin identity, **not the full spatial map or adjacency of pins**. It cannot identify where contact is moving or distinguish all deformation fields.

## 7. Classifier and learning algorithm

Use 12 inputs, 16 LIF hidden neurons and one linear output. At each step:

\[
v_t=0.875u_{t-1}+Wf_t+b,
\quad z_t=\mathbf1[v_t\ge1],
\quad u_t=v_t-z_t.
\]

Reset u=0 at each window. At the end of T steps:

\[
\ell=w^\top\left(\frac1T\sum_t z_t\right)+c,
\quad p=\sigma(\ell).
\]

The output decision is p≥τ, with τ selected on validation data. p is a model score, not established calibrated probability. There is no compulsory alarm merely because C or V is large.

Training uses weighted binary cross entropy, with positive weight n_non-slip/n_slip computed on training data. The hard spike's surrogate derivative is 1/(1+|v−1|)². The subtractive reset spike is detached in the backward calculation; the forward pass still resets normally. W,b,w,c all learn through explicit backpropagation through time.

Adam uses β₁=.9, β₂=.999, ε=10⁻⁸ and global gradient norm clipping at 5. Batch size is 64. Maximum 18 epochs, patience 4, best validation-loss checkpoint restored. The initial learning rate is .005. The threshold grid is .05,.06,…,.95; maximize validation balanced accuracy, breaking ties by proximity to .5, then lower threshold. Seeds control initialization and mini-batch order. Trial split seeds must be handled by the original data loader.

This is an ordinary LIF with a candidate input representation. The new method does not invent surrogate gradients, Adam, LIF, window pooling or normalization.

## 8. Computational budget and temporal support

Learned parameters: 12×16+16+16+1=225. Float32 learned-parameter storage would be 900 bytes, excluding scale, threshold, state, buffers and runtime. The reference actually uses NumPy float64, so its parameter arrays occupy 1,800 data bytes plus Python/array overhead. No quantized implementation is supplied.

A six-input LIF with 28 hidden neurons also has 6×28+28+28+1=225 parameters, providing an equal-parameter control. Equal parameter counts are not equal compute costs. The original 6→16→1 network has 129 parameters, so CT-LIF is larger.

Encoder work is O(18×3) per sample. Dense hidden projection takes 12×16=192 multiply-accumulates per encoded sample, plus state operations; output pooling/readout adds work. There is no measured hardware execution cost.

For streaming encoding, previous raw readings and previous activity fractions require 54+54 float values; previous gates and validity flags, model states, accumulators and scratch memory are additional. A naive 32-step 12-channel float32 feature buffer alone is 1,536 bytes. Sliding windows with reset neurons require replay or parallel window states; carrying one neuron state forever changes the model.

T encoded samples require T+2 raw samples: the first turnover needs two adjacent differences. For a 32-step model that is 34 raw samples, spanning 33 sampling intervals. Compared with the original 32-difference model, that is one extra preceding raw sample. Keep endpoint labels fixed and disclose this support difference. No physical slip lead time or real-time claim follows from window size.

## 9. Prior-art comparison and novelty boundary

Targeted public-web searches and primary-source inspection were performed on 8 October 2026. They are not an exhaustive scholarly or patent search. Some very recent results were only available as search abstracts; these remain review leads, not fully audited exclusions.

| Work inspected | Relevant established idea | Candidate distinction and remaining risk |
|---|---|---|
| Wenzhen Yuan's author project page, *Slip detection for general objects* [1] | Nonuniform marker motion and entropy can indicate slip | CT uses signed cancellation plus pin-identity turnover. Spatial nonuniformity itself is already known. |
| Jawale et al., *Learned Slip-Detection-Severity Framework…*, IROS 2024 [2], methodology inspected | Deformation magnitude, divergence/curl and their temporal changes; entropy controls | CT uses no spatial derivatives or coordinates. Both approaches summarize deformation dynamics, so a compact reformulation could be judged incremental. |
| Kang et al., *Boost event-driven tactile learning with location spiking neurons*, 2023 [3] | Spatial/temporal tactile SNNs and slip evaluation | CT keeps standard LIF and changes the encoder. Spatial SNN slip detection cannot be claimed as new. |
| *GradTac: Spatio-Temporal Gradient Based Tactile Sensing*, 2022 [4] | Spatial event aggregation and contour tracking for tactile motion/slip | CT uses pin activity distributions rather than contours. Tracking contact changes is already established. |
| Iskarous et al., *Invariant Neuromorphic Representations…*, 2025 [5] | Force/speed invariance in neuromorphic texture representations | CT's conditional gain invariance is algebraic and uses no force/speed input. Invariance itself is not novel. |
| Lu et al., *A Neuromorphic Incipient Slip Detection System using Papillae Morphology*, 2025 [6] | Spatial neuromorphic tactile slip classification | Different sensor/model, but further rules out a broad first-SNN-slip claim. |

**What was not identified in the inspected material:** the exact 12-channel construction combining signed common motion, gated cancellation and cancellation-weighted total-variation turnover of pin-indexed activity distributions, feeding this compact LIF. That observation is a limited search result—not proof of absence elsewhere, a legal opinion, or evidence that the combination is sufficiently inventive.

**Current defensible claim:** “We propose CT-LIF, a candidate compact tactile representation that couples normalized pin-motion cancellation with temporal redistribution of pin activity, and investigate its effect on cross-object slip recognition.”

**Do not claim:** “The world's first,” “not available anywhere,” “patented,” “proven novel,” “improved recall,” or “real-time ESP32 slip detection.” Those claims have not been established.

Before a publication novelty claim, inspect full texts and citations around marker entropy, vector-field features, normalized directional coherence, activity-distribution distances, tactile SNN encoding and spatial moment methods. Include conference papers, theses, preprints and relevant patents. Search by mechanisms and equivalent formulas, not just the new name. If a close equivalent exists, revise the contribution or report an engineering adaptation.

### Search log (representative queries actually used)

- tactile slip detection spatial normalization spiking neural network gain invariant
- tactile slip detection spatial coherence common mode deformation spiking neural network
- tactile slip detection marker motion cancellation normalized distribution temporal variation
- tactile slip "total variation" "distribution"
- "slip detection" "total variation distance"
- "slip detection" "polarity"
- "tactile" "cancellation" "slip"

Searches also surfaced *SlipSense: Multimodal Tactile Learning for Low-Latency and Generalized Slip Detection* (arXiv:2609.15910); its abstract page did not load successfully in this session. It remains a recent-work follow-up, not a cleared novelty comparison. Search engines can miss differently worded work.

## 10. Frozen exploratory evaluation proposal

Freeze this version and the following choices **before running any new test inference**. No new test scores were used to choose this candidate. However, historical held-out object errors already motivated the design, so those objects are no longer a fresh confirmatory test of the research process.

1. Reuse original whole-object and whole-trial partitions from the actual split manifests, not reconstructed guesses. Outer folds hold out brush, screwdriver and spool in turn. Retain the original two seeds 7/19 for direct exploratory comparisons.
2. First use T=32 and identical eligible endpoints for every method. Start with the historical endpoint grid and drop endpoints without the new encoder's required preceding raw samples. Do not cross trial boundaries. Score the final sample only. Re-run originals on the identical retained endpoint set.
3. Compare original LIF, six-input 28-neuron LIF, CT-LIF, CT mean-ReLU ANN and CT temporal ReLU ANN. Re-run the original 16-neuron mean representation as a reference. The supplied `mean` control is not the historical loader/normalizer.
4. Include C-only, independent-turnover (V instead of C×V), and entropy/change controls with the same 12-channel network size. Add simple threshold/logistic controls and close published representations for any strong research claim; these latter methods are not all implemented in this package.
5. The initial CT model is η=.02, rate=.005. For a bounded selection study, allow the prespecified rate grid {.005,.01} and gate grid {.005,.02,.05}. Compare encoders at each common setting as well as after validation selection. Mean controls have no gate parameter: disclose that their candidate budget differs. Do not present gate selection as an architecture-only comparison. Use the same two-rate budget for classifier comparisons on fixed CT features.
6. Choose checkpoints by validation loss and thresholds by validation balanced accuracy. Choose encoder hyperparameters by validation BA; ties prefer η=.02, then .005, then .05, followed by smaller learning rate. Record every fit. An expanded seed study is welcome if frozen in advance; it does not create additional independent objects.
7. Primary outcome: per-object slip recall and non-slip false-alarm rate, presented together. Also report object-macro BA, precision, F1, confusion matrices and window/trial counts. A recall increase caused simply by calling everything “slip” is a failure, not a successful result. To study alarm trade-offs, choose a separate operating threshold on validation with FPR≤10%, maximizing recall (ties prefer lower FPR then higher threshold); include the all-negative operating point and report its zero recall if no useful point exists. Report actual test FPR—validation constraints do not guarantee test constraints.
8. Evaluate fixed-gain perturbations at {.25,.5,1,2,4}, independently of object labels, with frozen scales and thresholds. Add training-scale-relative noise and pin-failure stress tests. These are diagnostics, not substitutes for natural unseen objects or sensor measurements.
9. Produce paired whole-trial bootstrap intervals within each object, sharing resampled trial IDs across methods/seeds. Do not treat overlapping windows or multiple seeds as independent objects. Disclose conditional uncertainty and all exploratory comparisons.
10. Seek an untouched dataset or additional objects/sessions for confirmation after method selection. If CT fails against the conventional temporal ANN, the finding may concern representation rather than spikes. If C-only or independent turnover matches CT, the proposed coupling lacks support. If nothing improves at a useful false-alarm rate, reject the hypothesis rather than rename the same model as successful innovation.

The executable tools implement training, encoding, basic split checks, individual-run metrics and prediction export. They do **not** automate the full grid, original project reproduction, whole-trial bootstrap, alarm-constrained threshold or fresh-data study.

## 11. Known failure cases

- Same-direction motion above the gate gives q=±1 and C=0 regardless of its amplitude; useful slip-strength information is discarded. An amplitude-retaining encoder is an important later control, not a claimed improvement here.
- Opposite-sign stationary noise can yield high cancellation. A gate based on global training activity may be too weak or too strong for a new object.
- Cancellation can be high without activity turnover; turnover can be high without cancellation. The product deliberately ignores the latter in its last three channels, which may hurt performance.
- Changes in direction with unchanged absolute pin activity produce V=0. Full spatial geometry and per-pin direction history are not retained.
- Grip changes and contact transitions may imitate slip. Weak true slip below the gate is attenuated; the method is not guaranteed to fix spool recall.
- Differentiation amplifies noise. Irregular timestamps are still processed by sample steps, as in the existing study; no time-rate invariance is claimed.
- A larger model or longer input support can explain gains. Parameter-matched and horizon-matched controls are required.
- Float64 host success does not establish float32/int8 equivalence, embedded fit, speed, power savings, closed-loop stability or physical sensing validity.

## 12. Deliverables and next required inputs

Completed: mathematical specification, prior-art boundary, working NumPy implementation, LIF and conventional training paths, ablation encoders, split checks, checkpoint evaluation and synthetic verification.

Needed for a measured project result: original source archive, raw per-pin dataset or verified loader, exact trial split manifests and baseline configurations/checkpoints. These were not among the three attachments. They are needed to connect this candidate to the actual project without inventing data or changing the historical experiment silently.

The original reported results remain historical evidence. CT-LIF currently has **no measured real-data accuracy**.

## References

1. Wenzhen Yuan, author project page, *Slip detection for general objects*: https://people.csail.mit.edu/yuan_wz/slip-detection.htm
2. Jawale et al., *Learned Slip-Detection-Severity Framework using Tactile Deformation Field Feedback for Robotic Manipulation*: https://arxiv.org/html/2411.07442v1
3. Kang et al., *Boost event-driven tactile learning with location spiking neurons*: https://pmc.ncbi.nlm.nih.gov/articles/PMC10160479/
4. *GradTac: Spatio-Temporal Gradient Based Tactile Sensing*: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.898075/full
5. Iskarous et al., *Invariant Neuromorphic Representations of Tactile Stimuli Improve Robustness of a Real-Time Texture Classification System*: https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202401078
6. Lu et al., *A Neuromorphic Incipient Slip Detection System using Papillae Morphology*: https://arxiv.org/abs/2509.09546

Project evidence supplied by the user: `ADVANCED_REPORT.md`, `FOLLOWUP_REPORT.md`, `RESEARCH_SCOPING.md`, dated 7 October 2026 where stated. Source reports are not included or altered in this candidate package.
