#include "../firmware/esp32_advanced_replay/advanced_runtime.h"
#include "../firmware/esp32_advanced_replay/generated_advanced_model.h"
#include <assert.h>
#include <stdio.h>
using namespace advanced_runtime;

void dynamics() {
  Model m = {};
  m.kind = ALIF; m.beta = .875f; m.rho = .95f;
  m.threshold = 1; m.adaptStrength = .4f; m.inputBias[0] = 1.2f;
  float x[Inputs] = {};
  State s; s.reset();
  assert(s.step(x, m, FloatWeights) == 1);
  assert(fabsf(s.voltage[0] - .2f) < 1e-6f && s.adaptation[0] == 1);
  assert(s.step(x, m, FloatWeights) == 0);
  assert(fabsf(s.voltage[0] - 1.375f) < 1e-6f);
  assert(fabsf(s.adaptation[0] - .95f) < 1e-6f);
  assert(s.step(x, m, FloatWeights) == 1);
  assert(fabsf(s.voltage[0] - 1.023125f) < 1e-6f);
  assert(fabsf(s.adaptation[0] - 1.9025f) < 1e-6f);
  assert(s.counts[0] == 2);
  m.kind = ANN; m.outputWeights[0] = 1; m.outputQ[0] = 10; m.outputScale = .1f;
  s.reset();
  for (unsigned t = 0; t < 3; ++t) assert(s.step(x, m, FloatWeights) == 0);
  assert(fabsf(s.score(m, FloatWeights) - 1.2f) < 1e-6f);
  assert(fabsf(s.score(m, Int8Weights) - 1.2f) < 1e-6f);
  assert(s.voltage[0] == 0 && s.counts[0] == 0);
  s.reset(); assert(s.samples == 0 && s.hiddenSum[0] == 0);
}

void fixtureParity() {
  for (unsigned index = 0; index < advanced_model::MODEL_COUNT; ++index) {
    const Model &m = advanced_model::MODELS[index];
    for (unsigned p = 0; p < 2; ++p) {
      const auto precision = static_cast<Precision>(p);
      const Expected &e = p ? advanced_model::EXPECTED_Q[index] : advanced_model::EXPECTED_FLOAT[index];
      State s; s.reset();
      for (unsigned t = 0; t < advanced_model::TRACE_LENGTH; ++t)
        s.step(advanced_model::TRACE[t], m, precision);
      for (unsigned h = 0; h < Hidden; ++h) {
        assert(s.counts[h] == e.counts[h]);
        assert(fabsf(s.voltage[h] - e.voltage[h]) < 1e-4f);
        assert(fabsf(s.adaptation[h] - e.adaptation[h]) < 1e-4f);
        assert(fabsf(s.hiddenSum[h] - e.hiddenSum[h]) < 1e-4f);
      }
      const float score = s.score(m, precision);
      assert(fabsf(score - e.score) < 1e-4f);
      assert((score >= m.decisionThreshold) == (e.score >= m.decisionThreshold));
      printf("host_reference model=%s precision=%s score=%.9g expected=%.9g decision=%d\n",
             m.id, p ? "int8_weights_float_state" : "float32", score, e.score, score >= m.decisionThreshold);
    }
  }
}

int main() {
  dynamics(); fixtureParity();
  puts("PASS host_reference advanced dynamics/reset, Python float/int8-weight fixtures, decisions.");
  puts("No ESP32 timing, energy, or live physical slip evidence.");
}
