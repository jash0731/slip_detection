#ifndef GRIPPER_SNN_RUNTIME_H
#define GRIPPER_SNN_RUNTIME_H

#include <stdint.h>
#include <math.h>

namespace gripper_runtime {
enum class Mode { Dense, Sparse };

// Connected feedforward LIF hidden layer; linear readout, not a spiking output.
// No allocation, task pinning, GPIO, training, or platform dependencies.
template <unsigned Inputs, unsigned Hidden> struct Network {
  float voltage[Hidden];
  uint32_t counts[Hidden];
  uint32_t samples;

  void reset() {
    samples = 0;
    for (unsigned h = 0; h < Hidden; ++h) {
      voltage[h] = 0.0f;
      counts[h] = 0;
    }
  }

  static unsigned encode(float sample, const float (&thresholds)[Inputs],
                         uint8_t (&events)[Inputs]) {
    unsigned active = 0;
    const float magnitude = fabsf(sample);
    for (unsigned i = 0; i < Inputs; ++i) {
      events[i] = magnitude >= thresholds[i] ? 1 : 0;
      active += events[i];
    }
    return active;
  }

  unsigned step(const uint8_t (&events)[Inputs],
                const float (&weights)[Inputs][Hidden], float leak,
                float threshold, float resetAmount, Mode mode) {
    for (unsigned h = 0; h < Hidden; ++h) voltage[h] *= leak;
    // Preserve input order in both modes. Sparse skips only absent input MACs;
    // all neurons still decay and test threshold each tick.
    for (unsigned i = 0; i < Inputs; ++i) {
      if (mode == Mode::Sparse && !events[i]) continue;
      for (unsigned h = 0; h < Hidden; ++h)
        voltage[h] += static_cast<float>(events[i]) * weights[i][h];
    }
    unsigned spikes = 0;
    for (unsigned h = 0; h < Hidden; ++h) {
      if (voltage[h] >= threshold) {
        voltage[h] -= resetAmount; // At most one spike per tick.
        ++counts[h];
        ++spikes;
      }
    }
    ++samples;
    return spikes;
  }

  float score(const float (&readout)[Hidden], float bias) const {
    float value = bias;
    if (!samples) return value;
    for (unsigned h = 0; h < Hidden; ++h)
      value += readout[h] * (static_cast<float>(counts[h]) / samples);
    return value; // Raw logistic logit; score >= 0 is the binary decision.
  }
};
} // namespace gripper_runtime
#endif
