# Experiment A: ESP32 GPIO loopback

This sketch produces **hardware electrical measurements**, not simulation or
mechanical vibration/slip measurements. Experiments B (filtering) and C (core
placement under load) remain future work.

## Wiring (v1)

Use a **classic dual-core ESP32** development board, such as an ESP32-WROOM-32
DevKit. The sketch deliberately rejects single-core and other ESP32 families;
GPIO25/26 and the two-core schedule are specific to this first setup.

```text
One ESP32, USB powered
GPIO25 (stimulus output) ---- 1 kΩ resistor ---- GPIO26 (input)
```

Both pins share the board's ground. Disconnect the SW-420 output while this link
is installed; no sensor, external pulse generator, or LED is needed. Use the
GPIO numbers on the board pinout, not physical header positions. ESP32 GPIO
uses 3.3 V logic; do not connect either pin to 5 V. A startup HIGH/LOW check
stops the run if the loopback is absent or stuck.

## Upload and collect

1. Install Arduino IDE. Add
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json` to
   **Preferences → Additional Boards Manager URLs**, then install **esp32 by
   Espressif Systems** in Boards Manager (verified with **3.3.12**). See the
   [official installation guide](https://docs.espressif.com/projects/arduino-esp32/en/latest/installing.html).
2. Open `firmware/esp32_acquisition/esp32_acquisition.ino` (keep the matching
   folder name). Select your classic ESP32 board, typically **ESP32 Dev Module**,
   and its USB port. Use **Arduino Runs On → Core 1** if that option is shown.
   No extra Arduino libraries or PlatformIO are required.
3. Upload. If it stays at “Connecting”, hold **BOOT** until uploading begins,
   then release it. Open Serial Monitor at **115200 baud**, reset the board if
   needed, and wait for “Send s to start”.
4. Send `s` once (any line-ending setting works). Keep the monitor open for
   roughly one minute, until `# COMPLETE: 500 hardware target trials` appears.
   Copy the CSV header and all 500 rows into `hardware_results/<run-name>.csv`.
   Omit lines beginning with `#`; the analysis tool also tolerates these lines.
   Reset and send `s` for another run. Do not combine runs into one CSV.
5. Save a companion `<run-name>.md` with board/module name, Arduino IDE and
   **ESP32 board package version**, board menu settings (including CPU clock),
   code commit or archived sketch, wiring/photo, date, and any anomalies. Change
   `CODE_VERSION` and `WIRING_VERSION` when their corresponding setup changes.
6. From this directory, run:
   `python3 analyze_hardware.py hardware_results/<run-name>.csv`.
   It validates the raw fields and reports counts, miss rate, and detected-trial
   median, nearest-rank p95, and maximum latency. It refuses simulation CSV.

## What is measured

- 1, 5, 10, and 20 ms polling, then a `RISING` GPIO interrupt, all with the
  **immediate** rule. Each configuration has 100 target trials: five requested
  widths (250, 1000, 3000, 8000, 15000 µs) × 20 deterministic phase offsets
  distributed across 20 ms, including sub-ms offsets. The planned relative
  schedules and their order are identical for all methods. Actual edges can
  differ due to scheduling; preserve those differences in the log.
- A queued stimulus task on core 0 generates the pulse independently of the
  acquisition loop on core 1. Polling uses absolute timestamps and busy waits;
  overdue ticks are skipped, never replayed as catch-up samples. Interrupt
  mode timestamps the first rising-edge ISR while the loop yields. This is a
  baseline without an added CPU workload; acquisition itself still costs CPU.
- The input is idle LOW and active HIGH. Each trial starts at `epoch_us`, the
  requested rising call is `epoch_us + 40000 + phase_us`, and the decision
  deadline is that requested rise + requested width + 40000 µs. Capture closes
  at that deadline. Late generator edges and poll lateness/skips are flagged
  in `notes`; inspect those before interpreting differences.
- `first_pulse_us` is the timestamp immediately before the HIGH GPIO write.
  `pulse_schedule_us` is JSON `[[actual_high_call_us,actual_low_call_us]]`, using
  absolute microseconds since boot and a half-open active interval. The polling
  decision timestamp follows the first HIGH read; the interrupt decision
  timestamp is taken at ISR entry. `latency_us = decision_us - first_pulse_us`.
  These calls include instrumentation, GPIO/API overhead, and scheduling;
  they are not oscilloscope edge timestamps or mechanical slip latency.
- All 15 columns match `EXPERIMENT.md`. A miss retains its pulse timestamps
  and has empty decision/latency fields. An unexpected decision before the
  generated pulse is retained as `false_alarm` with a signed latency and a
  flag. No noise-only/idle trials are included, so this run cannot estimate a
  general false-alarm rate. `poll_interval_ms` is empty for interrupts.
- `run_id` includes `hardware`, code/wiring versions, chip ID, and a boot token.
  `core_config` records actual generator/acquisition/decision core IDs (`-1`
  means no decision). `load`, `parameters`, and `notes` record radio state,
  settings, priorities, CPU frequency, and timing anomalies. Logging happens
  only after acquisition and drains before the next trial; idle gaps allow
  watchdog tasks to run. Wi-Fi/Bluetooth are never started.

The timebase uses Espressif's
[`esp_timer_get_time()`](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/system/esp_timer.html#obtaining-current-time),
and GPIO acquisition uses the
[Arduino ESP32 GPIO API](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/gpio.html).
Core placement and short busy waits do not guarantee timing isolation. Do not
substitute ideal simulated interrupt latency for measured ISR latency.

## Results are separate

`simulation_results.csv` is generated only by `simulate.py` and remains clearly
**SIMULATED**. Place real serial captures only in `hardware_results/`. No hardware
result CSV is supplied with the sketch; hardware behavior must be checked on a
wired board. A host/toolchain check is not a hardware run.

Development checks: the sketch compiled for `esp32:esp32:esp32` (ESP32 Dev
Module) with Arduino ESP32 **3.3.12** and warnings enabled, using 280588 bytes
of flash and 22860 bytes of static RAM. Capture analysis has five passing
synthetic-fixture tests (`python3 -m unittest discover -s tests -v`). These are
build/parser checks only; no board was uploaded or measured during development.
