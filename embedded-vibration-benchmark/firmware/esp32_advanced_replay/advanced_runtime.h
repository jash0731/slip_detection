#pragma once
#include <stdint.h>
#include <math.h>

namespace advanced_runtime {
constexpr unsigned Inputs = 6;
constexpr unsigned Hidden = 16;
enum Kind { LIF = 0, ALIF = 1, ANN = 2 };
enum Precision { FloatWeights = 0, Int8Weights = 1 };

struct Model {
  const char *id;
  const char *hash;
  Kind kind;
  float inputWeights[Hidden][Inputs];
  float inputBias[Hidden];
  float outputWeights[Hidden];
  float outputBias;
  float decisionThreshold; // Validation-selected raw-logit threshold.
  int8_t inputQ[Hidden][Inputs];
  int8_t outputQ[Hidden];
  float inputScale;
  float outputScale;
  float beta;
  float rho;
  float threshold;
  float adaptStrength;
};

struct Expected {
  uint32_t counts[Hidden];
  float voltage[Hidden];
  float adaptation[Hidden];
  float hiddenSum[Hidden];
  float score;
};

struct State {
  float voltage[Hidden];
  float adaptation[Hidden];
  float hiddenSum[Hidden];
  uint32_t counts[Hidden];
  uint32_t samples;

  void reset() {
    samples = 0;
    for (unsigned h = 0; h < Hidden; ++h) {
      voltage[h] = adaptation[h] = hiddenSum[h] = 0;
      counts[h] = 0;
    }
  }

  unsigned step(const float (&x)[Inputs], const Model &model, Precision precision) {
    unsigned spikes = 0;
    for (unsigned h = 0; h < Hidden; ++h) {
      float current = 0;
      for (unsigned i = 0; i < Inputs; ++i) {
        const float w = precision == Int8Weights ? model.inputQ[h][i] * model.inputScale
                                                 : model.inputWeights[h][i];
        const float product = w * x[i];
        current += product;
      }
      current += model.inputBias[h];
      if (model.kind == ANN) {
        const float activation = current > 0 ? current : 0;
        hiddenSum[h] += activation;
        continue;
      }
      const float decayed = model.beta * voltage[h];
      voltage[h] = decayed + current;
      const float threshold = model.threshold +
        (model.kind == ALIF ? model.adaptStrength * adaptation[h] : 0);
      const bool spike = voltage[h] >= threshold;
      if (spike) {
        voltage[h] -= threshold;
        ++counts[h];
        ++spikes;
        hiddenSum[h] += 1;
      }
      if (model.kind == ALIF) {
        const float decayedAdaptation = model.rho * adaptation[h];
        adaptation[h] = decayedAdaptation + (spike ? 1 : 0);
      }
    }
    ++samples;
    return spikes;
  }

  float score(const Model &model, Precision precision) const {
    float value = 0;
    if (!samples) return model.outputBias;
    for (unsigned h = 0; h < Hidden; ++h) {
      const float w = precision == Int8Weights ? model.outputQ[h] * model.outputScale
                                               : model.outputWeights[h];
      const float product = w * (hiddenSum[h] / samples);
      value += product;
    }
    return value + model.outputBias;
  }
};
} // namespace advanced_runtime
