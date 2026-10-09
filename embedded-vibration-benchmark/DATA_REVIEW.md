# Dataset decision: robotic-gripper slip remains the application

Reviewed 6 October 2026. This review selects an accessible public recording for an
exploratory host comparison and ESP32 replay demonstration. No model accuracy,
ESP32 execution time, piezo sensor transfer, or closed-loop grip performance is
claimed here.

## Selected first dataset

Use the authors' [ARQ-CRISP 2021 robotic pick-and-place dataset](https://github.com/ARQ-CRISP/slip_detection_dataset_2021).
The repo describes an EZGripper/UR5 with an 18-pin uSkin sensor, three channels per
pin, nominal 180 Hz sampling, three objects and 30 grasps per object. It supplies
manually annotated perceived-slip labels and a BSD-3-Clause license. Retain the
license and cite Zenha et al., *Tactile Slip Detection in the Wild Leveraging
Distributed Sensing of both Normal and Shear Forces*, IROS 2021,
[DOI:10.1109/IROS51168.2021.9636602](https://doi.org/10.1109/IROS51168.2021.9636602).

For the first runnable study, the smallest file, brush, was actually downloaded,
checksum verified and inspected. Its 30 grasp recordings contain 41,707 samples
with finite values and strictly increasing timestamps within every recording.
The local source is `data/public/ARQ_CRISP_2021/slipDataset_brush_tactile.h5`;
its pinned commit is `a536ee3301af0566218782b00de334b6a6630979`, file size
41,105,856 bytes, SHA256
`a2aef56bf4cfc5e2d17ae09772295cb207c733e3a73059913939ba5a7a1e2139`.
Raw and prepared binary files are ignored by Git; the fetcher recreates them.

The audited schemas and label meanings match the [authors' data documentation](https://github.com/ARQ-CRISP/slip_detection_dataset_2021/blob/main/data/README.md).
Each trial has `(18,3,N)` tactile data, `N` timestamps in milliseconds,
`N` slip labels, and an additional change-label array. Channel axes are x/y shear
and z normal. Labels 1 and 5 indicate perceived slip; 2, 3 and 6 are release,
grasp and other events. Label 0 mixes not-held and held-static non-slip;
label 4 is held non-slip during arm motion. Our audit corrects the source README's
informal label-count wording by inspecting actual integer tags rather than
assuming its example counts.

### Prefer a held-state comparison

For the primary classification comparison, use label 4 versus label 5, both
representing a held object during arm motion. This reduces the shortcut of
separating free space from grasped objects. The broader 0/4 versus 1/5 mapping is
available as a secondary task, explicitly labelled as such.

| Grasp pose | Held non-slip, label 4 | Held slip, label 5 |
| --- | ---: | ---: |
| Pose 1 | 2,247 | 2,495 |
| Pose 2 | 5,666 | 1,195 |
| Pose 3 | 705 | 1,898 |
| Total | 8,618 | 5,588 |

Only 10 of the 30 recordings contain label 4. Thus thousands of samples do not
represent thousands of independent negative examples. Reset model state between
recordings, preserve causal time context even for unscored labels, and split
whole recordings. A held-out-pose test (train/validate on poses 1/2; final test on
pose 3) is possible, but its object remains the same brush. Never call that
cross-object generalization. Fit scaling and encoder parameters on training data
only. Independent recording-level uncertainty is more appropriate than treating
every frame as an independent replicate.

The actual frame intervals span approximately 0.46–15.33 ms, with typical medians
near 5.5 ms. Report replay compute time separately from sampling delay and
human annotation uncertainty. These labels do not independently identify the
first incipient-slip instant, so they cannot establish a sub-5 ms physical
response or prediction before gross slip. Public-recording classification is a
first evidence stage toward incipient-slip detection for robotic grippers.

## Practical tools and evidence

From the project root:

```sh
python3 data/fetch_arq_crisp.py
python3 data/audit_arq_crisp.py data/public/ARQ_CRISP_2021/slipDataset_brush_tactile.h5 --output data/public/ARQ_CRISP_2021/audit.json
python3 data/prepare_arq_crisp.py
```

Python with NumPy and h5py is required. Download uses the GitHub media URL for
real Git-LFS contents, not the tiny raw pointer file. The fetcher verifies the
pinned SHA256 and retains upstream license/readmes and provenance. Audit checks
shape, finiteness, timing, label counts and slip runs. The preparation script
exports `data/arq_brush_trials.npz` with raw float32 inputs, full-task `y`, primary
`y_held`, original labels, relative timestamps, trial groups and pose identifiers.
See `data/README.md` for the complete interface. No normalized inputs or trained
weights are hidden in the export.

For a sensor-free hardware demo, choose a short held-out recording segment,
export inputs as Arduino flash arrays, and run the same causal model state
updates on the ESP32. This measures our board's inference/replay behavior on
public robotic-gripper signals. Record whether replay follows original timing
or runs as fast as possible. It does not demonstrate physical sensing or
closed-loop grip-force adjustment. uSkin normal/shear readings are not piezo
voltages; success on this dataset cannot establish transfer to a cheap piezo
sensor without later evidence.

## First host pipeline audit

The current exploratory pipeline uses nonoverlapping 32-sample windows and
retains only windows wholly labelled 4 or wholly labelled 5. It yields 306
training windows from poses 1/2 (223 negative, 83 positive), and 55 test windows
from pose 3 (13 negative, 42 positive). Those test negatives arise from only
three independent grasp trials. The split shares no trial identifiers across
training and test, and normalization uses training poses only.

Because homogeneous windows are selected using their ground-truth labels,
reported accuracy is conditional on sustained labelled held states. Onset and
ambiguous windows are excluded; this is not unrestricted online detection
accuracy. Filter and neuron states reset at each window in this first pipeline.
Thus it is a window-classification scaffold, rather than an evaluation of
persistent streaming-state decisions. The causal raw first difference preserves
original sample order and preceding-sample context without removing ignored gaps.

The initial Arduino parity fixture is the first positive held-out window in
fixed recording order: `GGCNN2/brush/Pose3/Exp1`, samples 544–575. It is selected
by label and ordering, without screening prediction correctness. One fixture is
for reproducible numerical/board checks, not estimation of classifier accuracy
or representative performance. Its scalar inputs activate almost all four
encoder thresholds, so activity-dependent speed conclusions must use additional
controlled workloads rather than generalizing from that trace.

## Alternatives inspected, not downloaded

The [NeuTouch slip dataset](https://clear-nus.github.io/visuotactile/download.html)
has 50 rotational-slip and 50 stable recordings, tactile events, optional vision,
and OptiTrack-derived motion information. Its 300 MB lite archive could provide
an event-native extension. Its detailed event/ground-truth alignment and dataset
license require inspection before adoption; the site alone does not establish
incipient onset or inexpensive sensor transfer.

The [PapillArray incipient-slip repository](https://github.com/RedmondLabUCD/Incipient-Slip-Detection-PapillArray)
and linked [Zenodo archive](https://zenodo.org/records/13228084) are closer to the
incipient-slip objective. The linked archive is approximately 1.5 GB, and supplied
merged training data are already augmented. Independent split provenance and
underlying time labels must be audited before using it to support onset claims.
Its specialized optical sensor also differs from piezo hardware.

The [DENSE 2025 dataset](https://zenodo.org/records/17336134) offers broader grasp
and object variation, with a roughly 158 MB HDF5 archive. Its description uses
sequence-wide approximate slip/stable labels, which do not supply precise onset
latency. It is a potential cross-object classification follow-up after the first
pipeline works. Texture-only and object-recognition datasets were not selected
because those labels would change the application.

## Next actions

1. Freeze primary held-state task and recording-level partitions before tuning.
2. Compare conventional and spiking models with the same causal input information.
3. Export a held-out trace and learned parameters for board replay; verify host/C
   state parity before physical timing measurements.
4. Expand to additional objects using the other two pinned source files if time
   permits; the source repository lists them, but they are not yet downloaded.
5. Keep incipient-slip prediction and closed-loop grip response as explicit
   validation questions, not conclusions drawn from coarse slip labels.
