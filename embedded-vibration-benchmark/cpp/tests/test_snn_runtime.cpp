#include "../../firmware/esp32_snn_replay/snn_runtime.h"
#include "../../firmware/esp32_snn_replay/generated_model.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

using Net = gripper_runtime::Network<gripper_model::INPUTS, gripper_model::HIDDEN>;

void demoParity() {
  Net dense, sparse;
  dense.reset();
  sparse.reset();
  for (unsigned t = 0; t < gripper_model::TRACE_LENGTH; ++t) {
    uint8_t events[gripper_model::INPUTS];
    Net::encode(gripper_model::TRACE[t], gripper_model::INPUT_THRESHOLDS, events);
    const auto a = dense.step(events, gripper_model::WEIGHTS, gripper_model::LEAK,
                              gripper_model::THRESHOLD, gripper_model::RESET,
                              gripper_runtime::Mode::Dense);
    const auto b = sparse.step(events, gripper_model::WEIGHTS, gripper_model::LEAK,
                               gripper_model::THRESHOLD, gripper_model::RESET,
                               gripper_runtime::Mode::Sparse);
    assert(a == b);
    for (unsigned h = 0; h < gripper_model::HIDDEN; ++h) {
      assert(dense.counts[h] == sparse.counts[h]);
      assert(dense.voltage[h] == sparse.voltage[h]);
    }
  }
  for (unsigned h = 0; h < gripper_model::HIDDEN; ++h) {
    assert(dense.counts[h] == static_cast<uint32_t>(gripper_model::EXPECTED_COUNTS[h]));
    assert(fabsf(dense.voltage[h] - gripper_model::EXPECTED_VOLTAGE[h]) < 1e-5f);
  }
  const float score = dense.score(gripper_model::READOUT, gripper_model::BIAS);
  assert(fabsf(score - gripper_model::EXPECTED_SCORE) < 1e-5f);
  dense.reset();
  assert(dense.samples == 0);
  for (unsigned h = 0; h < gripper_model::HIDDEN; ++h) {
    assert(dense.counts[h] == 0);
    assert(dense.voltage[h] == 0.0f);
  }
}

void dynamicsAndEncoder() {
  gripper_runtime::Network<2, 1> n;
  const float thresholds[2] = {0.2f, 0.5f};
  uint8_t events[2];
  assert(n.encode(-0.5f, thresholds, events) == 2);
  assert(n.encode(0.0f, thresholds, events) == 0);
  const float weights[2][1] = {{0.75f}, {0.5f}};
  n.reset();
  events[0] = 1; events[1] = 0;
  assert(n.step(events, weights, 0.5f, 1.0f, 1.0f, gripper_runtime::Mode::Sparse) == 0);
  assert(n.voltage[0] == 0.75f);
  assert(n.step(events, weights, 0.5f, 1.0f, 1.0f, gripper_runtime::Mode::Sparse) == 1);
  assert(n.voltage[0] == 0.125f);
  events[0] = 0;
  assert(n.step(events, weights, 0.5f, 1.0f, 1.0f, gripper_runtime::Mode::Sparse) == 0);
  assert(n.voltage[0] == 0.0625f); // Silence still leaks.
  assert(n.counts[0] == 1);
}

void activityParity() {
  const float amplitudes[] = {0.0f, 0.2f, 0.4f, 0.7f, 1.2f};
  for (unsigned a = 0; a < 5; ++a) {
    Net dense, sparse;
    dense.reset(); sparse.reset();
    uint8_t events[gripper_model::INPUTS];
    assert(Net::encode(amplitudes[a], gripper_model::INPUT_THRESHOLDS, events) == a);
    for (unsigned t = 0; t < 10000; ++t) {
      assert(dense.step(events, gripper_model::WEIGHTS, gripper_model::LEAK,
                        gripper_model::THRESHOLD, gripper_model::RESET, gripper_runtime::Mode::Dense) ==
             sparse.step(events, gripper_model::WEIGHTS, gripper_model::LEAK,
                         gripper_model::THRESHOLD, gripper_model::RESET, gripper_runtime::Mode::Sparse));
      for (unsigned h = 0; h < gripper_model::HIDDEN; ++h) {
        assert(dense.voltage[h] == sparse.voltage[h]);
        assert(dense.counts[h] == sparse.counts[h]);
      }
    }
  }
}

int main() {
  dynamicsAndEncoder();
  demoParity();
  activityParity();
  puts("PASS host_reference: dense/sparse step parity, exported Python fixture, reset, decay, encoder, five activity levels.");
  puts("No hardware timing or physical-slip measurements were performed.");
}
