# Research upgrade: trained adaptive spiking inference

Design and targeted literature review, 7 October 2026. This extends the initial
fixed-hidden/readout experiment; it does not replace its saved results.
The robotic-gripper application and USB-only course hardware demonstration remain.
No additional components or mechanical gripper are required.

## What makes the extension stronger

The first comparison established an executable pipeline but had three major
limits: one object, fixed hidden weights, and label-selected homogeneous windows.
The extension addresses those limits with all available object recordings,
end-to-end training, and causal windows scored by the label at their final sample.
Sophistication here means controlled comparisons and credible evidence, rather
than adding a large architecture that cannot be evaluated before the review.

## Additional primary sources

| Source | Methodological lesson | Decision for this project |
| --- | --- | --- |
| [Neftci, Mostafa and Zenke, 2019, Surrogate Gradient Learning in Spiking Neural Networks](https://arxiv.org/abs/1901.09948) | Hard threshold spikes can use a surrogate derivative during optimization. The inference forward pass can still use discrete spikes. | Train the input and readout weights of a small LIF network offline, export its hard-spiking inference, and verify parameters actually changed. |
| [Bellec et al., NeurIPS 2018, Long short-term memory and Learning-to-learn in networks of spiking neurons](https://papers.neurips.cc/paper_files/paper/2018/hash/c203d8a151612acf12457e4d67635a95-Abstract.html) | Adaptive neuronal thresholds can provide longer temporal state; the paper studies trained recurrent networks. | Test an adaptive-threshold LIF variant with an extra adaptation state. Our small feed-forward variant is inspired by adaptation, not a reproduction of their full recurrent LSNN. |
| [Taunyazov et al., RSS 2020, Event-Driven Visual-Tactile Sensing and Learning for Robots](https://arxiv.org/abs/2009.07083) | Event-based tactile/visual SNN perception and rotational-slip evaluation are established prior work. | Retain slip as the application; do not claim that applying an SNN to slip is novel. Our measurements concern ordinary microcontroller execution and representation trade-offs. |
| [Grover et al., 2022, Learning to Detect Slip with Barometric Tactile Sensors and a Temporal Convolutional Neural Network](https://arxiv.org/abs/2202.09549) | Conventional temporal models are credible slip detectors and must not be omitted from comparisons. | Include a trained conventional network plus a simple causal-feature control; do not set the SNN against only weak untuned rules. |
| [Espressif ESP-DL quantization documentation](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_quantize_model.html) | Weight precision, activation precision and platform-specific quantization are separate choices. | Initially test int8 weight storage with floating-point neuron states in our custom runtime. It is not ESP-DL deployment, fully integer inference, or guaranteed acceleration. |
| [Espressif ESP-NN](https://github.com/espressif/esp-nn) | Optimized conventional neural kernels depend on the chip and operations. | Use a matched straightforward C++ comparison first; disclose that it does not establish superiority over every optimized ANN runtime. |

Sources were assessed from their accessible abstracts/documentation and the
available RSS text. This is a targeted review, not an exhaustive novelty search.
Prior-work quality/energy numbers do not become measurements of our board.

## Experimental contract

1. Preserve original raw timestamps, trial boundaries and ignored labels.
   Prepare causal normal/shear summaries rather than reducing all axes to one
   vibration magnitude. Fit all normalization on training trials only.
2. Hold out one whole object at a time. Within the remaining objects, reserve
   complete trials for validation. Record every trial identifier. Do not use
   the held-out object for thresholds, epoch selection or preprocessing.
3. Score causal 32-sample windows by their final sample's held
   state, rather than requiring future labels to be homogeneous. Earlier
   ignored/transition samples remain input context. Model state still resets at
   the beginning of a window; this is not a continuously stateful detector.
   The implemented stride is 16 samples: windows overlap and remain dependent,
   so uncertainty must retain whole trials rather than resampling windows.
4. Compare a simple trained causal-feature control, a trained ANN, a trained
   LIF SNN and an adaptive LIF SNN. ANN/LIF/ALIF share the input dimensions and
   hidden width. A readout-only LIF ablation can isolate hidden-weight learning.
5. Use a bounded, fixed training budget and validation-only selection. Record
   loss histories, seeds, best epochs, selected decision thresholds and model
   hashes. Label the extension exploratory: design was informed by the earlier
   brush experiment, not preregistered independently of all observations.
6. Quantize weights after training. Preserve floating-point biases and states,
   use explicit scales and rounding, and keep decision thresholds frozen.
   Measure changes in spikes, scores, decisions and test quality.
7. Export static parameters and the same held-out input fixture to C++/Arduino.
   Validate float and quantized inference against their respective reference
   traces. Quantized output need not equal float output; any disagreement is
   itself an outcome to report.

## Questions and primary outcomes

- **Learning ablation:** does training hidden connections help under identical
  representation, capacity, split and decision horizon? Changes between the
  old and new pipelines alone cannot isolate this cause.
- **Adaptation ablation:** does adaptive-threshold state improve balanced
  accuracy or false-alarm/recall trade-offs enough to justify its state and
  update cost relative to LIF?
- **Storage ablation:** what weight/scale bytes are saved, and what changes in
  test quality and numerical behavior follow? Smaller weight storage is not
  proof of smaller total firmware or faster inference.
- **Transfer:** how much do outcomes vary when the excluded object changes?
  Three objects support a limited cross-object study, not broad deployment
  generalization.

Balanced accuracy is the primary quality measure. Also report F1, confusion
counts and per-object outcomes. If uncertainty is estimated, resample whole
trials with paired predictions; adjacent windows are not independent replicates.
A pooled interval over these trials does not establish uncertainty over the
population of all possible objects. Do not convert a descriptive margin into a
formal noninferiority claim without adequate independent replication.

## Timing and evidence boundaries

The advanced board replay times prepared six-channel input, model updates and
window readout. Original raw tactile reduction is offline and excluded; memory
copy time is not physical sensing time. A 32-sample observation horizon is
approximately 178 ms at the source's nominal rate, even if computation is much
faster. The first advanced replay is a throughput experiment, not a demonstrated
live deadline or slip-onset latency.

Direct continuous inputs mean input accumulation is dense. Binary hidden spikes
do not automatically make this network fully event-driven. The original sparse
activity experiment remains a separate, explicitly defined optimization study.

Evidence categories stay separate: original public robotic recordings, our host
training/evaluation, synthetic host stress tests, and actual board captures.
Manual perceived-slip labels cannot establish prediction before physical onset.
The inexpensive vibration sensor's eventual live LED demonstration is an
acquisition demonstration; the trained uSkin model cannot be assumed to transfer
to that different sensor. No energy measurement or guaranteed timing is claimed.
