// Host computation measurements only, never ESP32 timing or energy evidence.
#include "../firmware/esp32_snn_replay/snn_runtime.h"
#include "../firmware/esp32_snn_replay/generated_model.h"
#include <chrono>
#include <stdio.h>

using Net = gripper_runtime::Network<gripper_model::INPUTS, gripper_model::HIDDEN>;
volatile double checksumSink = 0;
constexpr unsigned STEPS = 200000;

void run(float amplitude, gripper_runtime::Mode mode, unsigned trial, bool emit) {
  Net n;
  n.reset();
  uint8_t events[gripper_model::INPUTS];
  const unsigned active = Net::encode(amplitude, gripper_model::INPUT_THRESHOLDS, events);
  const auto start = std::chrono::steady_clock::now();
  for (unsigned t = 0; t < STEPS; ++t)
    n.step(events, gripper_model::WEIGHTS, gripper_model::LEAK,
           gripper_model::THRESHOLD, gripper_model::RESET, mode);
  const auto end = std::chrono::steady_clock::now();
  double checksum = 0;
  for (unsigned h = 0; h < gripper_model::HIDDEN; ++h)
    checksum += n.counts[h] + n.voltage[h];
  checksumSink += checksum;
  if (!emit) return;
  const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("host_reference,%s,%u,%s,%u,%u,%u,%lld,%.9g\n",
         gripper_model::MODEL_ID, trial,
         mode == gripper_runtime::Mode::Dense ? "dense" : "sparse",
         active, gripper_model::INPUTS, STEPS, static_cast<long long>(ns), checksum);
}

int main() {
  const float amplitudes[] = {0.0f, 0.2f, 0.4f, 0.7f, 1.2f};
  puts("# host_reference only; compiler/CPU dependent; no ESP32 or slip measurements.");
  puts("# Fixed synthetic encoder amplitudes; timing hidden dynamics only, excluding encoder/readout.");
  puts("source,model_id,trial,mode,active_inputs,input_channels,steps,duration_ns,checksum");
  for (float amplitude : amplitudes) {
    run(amplitude, gripper_runtime::Mode::Dense, 0, false);
    run(amplitude, gripper_runtime::Mode::Sparse, 0, false);
    for (unsigned trial = 1; trial <= 10; ++trial) {
      run(amplitude, trial % 2 ? gripper_runtime::Mode::Dense : gripper_runtime::Mode::Sparse, trial, true);
      run(amplitude, trial % 2 ? gripper_runtime::Mode::Sparse : gripper_runtime::Mode::Dense, trial, true);
    }
  }
}
