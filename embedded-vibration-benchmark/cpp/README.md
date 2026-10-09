Host correctness checks for the exact C++ kernel used by Arduino:

```sh
c++ -std=c++11 -O2 -Wall -Wextra -pedantic cpp/tests/test_snn_runtime.cpp -o /tmp/gripper-snn-test
/tmp/gripper-snn-test
```

Run from the project folder. These are `host_reference` checks, never hardware
results. They check every replay tick for dense/sparse equality, final values
against the exported Python fixture, encoder boundaries, silent decay, soft
reset, state reset, and 10,000-tick dense/sparse parity at each of five activity
levels. Inspect `generated_model.h` for the exported model and
trace provenance; public-data parity is not live physical-slip validation.

Do not use `-ffast-math`: reassociation can change threshold crossings.

Optional host activity sweep (0, 1, 2, 3, and 4 active encoder channels):

```sh
c++ -std=c++11 -O2 -Wall -Wextra -pedantic cpp/activity_benchmark.cpp -o /tmp/gripper-activity
/tmp/gripper-activity > /tmp/gripper-host-activity.csv
```

Ten paired repetitions per activity use 200,000 ticks, preallocated state,
warm-up, alternating mode order, and a volatile state checksum. Encoder and
readout are outside this timing interval. This measures the **host compiler and
CPU**, not ESP32 crossover, energy, or physical slip. Record compiler flags and
host platform before using the file in a report. Zero-activity runs still
update leak and threshold dynamics, and sparse skips the dense zero terms.
