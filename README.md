# Slip detection with CT-LIF

The latest implementation combines CT-LIF cancellation and turnover features with optimized mean inputs.

See [setup instructions](embedded-vibration-benchmark/CT_LIF_START_HERE.md) and [measured results](embedded-vibration-benchmark/CT_LIF_RESULTS.md).

Combined model: 69.19% balanced accuracy, 51.49% slip recall, 13.12% false-positive rate. Lower false alarms accompany reduced recall. Publication novelty is unverified.

Source code, research protocols and summaries are included here. Large datasets and detailed per-window prediction files are omitted from Git. The complete delivered CT_LIF_Tested_Project.zip includes those files; extract it into this checkout to reproduce the full saved-model audits. Downloaded raw data originate from https://github.com/ARQ-CRISP/slip_detection_dataset_2021.
