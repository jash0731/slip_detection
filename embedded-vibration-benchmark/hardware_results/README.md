# Hardware captures only

The first advanced replay measurement is `advanced-run-clean.csv`. It contains
validated text output from the connected ESP32-D0WD-V3. The original
`advanced-run.csv` is retained with reset boot bytes before its metadata header.

No measurements have been collected in this repository. Keep the full serial
output, including metadata, header and COMPLETE marker, for each independent run.

- Current trained-model comparison: complete 1,920-row capture; follow
  [ADVANCED_HARDWARE.md](../ADVANCED_HARDWARE.md) and use `analyze_advanced.py`.
- GPIO acquisition: complete 500-row capture; follow [HARDWARE.md](../HARDWARE.md)
  and use `analyze_hardware.py`.
- SNN trace replay: paired dense/sparse captures; follow
  [SNN_HARDWARE.md](../SNN_HARDWARE.md) and use `analyze_replay.py`.
- SNN controlled activity sweep: measured execution of synthetic activity levels;
  use its separate activity schema/analyzer, not the replay or GPIO analyzer.

Use names such as `replay-board1-boot1.csv`, and keep board/toolchain/code settings
alongside each capture. Board computation on prerecorded or synthetic input must
not be described as our own physical slip acquisition.

Do not copy `simulation_results.csv` here or label simulated/host-test output as
hardware evidence.
