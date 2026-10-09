// USB-only advanced recorded-input computation demo for robotic-gripper research.
#include <Arduino.h>
#include <esp_timer.h>
#include <esp_random.h>
#include <inttypes.h>
#include "advanced_runtime.h"
#include "generated_advanced_model.h"

constexpr unsigned TRIALS = 10;
constexpr char CODE_ID[] = "gripper-advanced-replay-v1";
advanced_runtime::State state;
struct StepRecord { int64_t preparationUs; int64_t inferenceUs; unsigned spikes; };
StepRecord records[advanced_model::TRACE_LENGTH];
char runId[96];
bool finished = false;

bool matchesFixture(unsigned index, advanced_runtime::Precision precision) {
  const auto &expected = precision == advanced_runtime::Int8Weights ?
                         advanced_model::EXPECTED_Q[index] : advanced_model::EXPECTED_FLOAT[index];
  for (unsigned h = 0; h < advanced_runtime::Hidden; ++h) {
    if (state.counts[h] != expected.counts[h] ||
        fabsf(state.voltage[h] - expected.voltage[h]) > 1e-4f ||
        fabsf(state.adaptation[h] - expected.adaptation[h]) > 1e-4f ||
        fabsf(state.hiddenSum[h] - expected.hiddenSum[h]) > 1e-4f) return false;
  }
  return fabsf(state.score(advanced_model::MODELS[index], precision) - expected.score) <= 1e-4f;
}

bool replay(unsigned index, advanced_runtime::Precision precision, unsigned trial, bool emit) {
  const auto &model = advanced_model::MODELS[index];
  state.reset();
  const int64_t begin = esp_timer_get_time();
  for (unsigned tick = 0; tick < advanced_model::TRACE_LENGTH; ++tick) {
    float input[advanced_runtime::Inputs];
    const int64_t start = esp_timer_get_time();
    for (unsigned i = 0; i < advanced_runtime::Inputs; ++i) input[i] = advanced_model::TRACE[tick][i];
    const int64_t prepared = esp_timer_get_time();
    const unsigned spikes = state.step(input, model, precision);
    const int64_t inferred = esp_timer_get_time();
    records[tick] = {prepared - start, inferred - prepared, spikes};
  }
  const int64_t readoutStart = esp_timer_get_time();
  const float score = state.score(model, precision);
  const int64_t end = esp_timer_get_time();
  if (!matchesFixture(index, precision)) {
    Serial.printf("# ERROR fixture mismatch model=%s precision=%s; incomplete capture.\n",
                  model.id, precision == advanced_runtime::Int8Weights ? "int8_weights_float_state" : "float32");
    return false;
  }
  if (!emit) return true;
  const char *name = precision == advanced_runtime::Int8Weights ? "int8_weights_float_state" : "float32";
  const bool decision = score >= model.decisionThreshold;
  const bool reference = advanced_model::EXPECTED_FLOAT[index].score >= model.decisionThreshold;
  Serial.printf("# trace model=%s precision=%s trial=%u elapsed_us=%" PRId64
                " readout_us=%" PRId64 " fixture_parity=pass\n", model.id, name,
                trial, end - begin, end - readoutStart);
  for (unsigned tick = 0; tick < advanced_model::TRACE_LENGTH; ++tick) {
    const StepRecord &r = records[tick];
    Serial.printf("hardware_advanced_replay,%s,%s,%s,%s,%s,%u,%s,%u,%" PRId64
                  ",%" PRId64 ",%" PRId64 ",%u,%.9g,%.9g,%d,%d,%d\n", runId,
                  model.id, model.hash, advanced_model::INPUT_ID, advanced_model::INPUT_HASH,
                  trial, name, tick, r.preparationUs, r.inferenceUs,
                  r.preparationUs + r.inferenceUs, r.spikes,
                  static_cast<double>(score), static_cast<double>(model.decisionThreshold),
                  decision, reference, decision != reference);
  }
  Serial.flush();
  delay(5);
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  snprintf(runId, sizeof(runId), "%s-%08" PRIx32, CODE_ID, esp_random());
  Serial.println("# SG-trained LIF/ALIF and matched ANN; recorded input only, no live slip sensing.");
  Serial.println("# Press s then Enter to run comparisons once; reset to repeat.");
}

void loop() {
  if (finished || !Serial.available()) { delay(10); return; }
  if (Serial.read() != 's') return;
  finished = true;
  Serial.printf("# code=%s chip=%s cpu_mhz=%u sdk=%s run_id=%s\n", CODE_ID,
                ESP.getChipModel(), static_cast<unsigned>(ESP.getCpuFreqMHz()), ESP.getSdkVersion(), runId);
  Serial.printf("# source=hardware_advanced_replay input_origin=public_recorded_tactile input=%s input_hash=%s\n",
                advanced_model::INPUT_ID, advanced_model::INPUT_HASH);
  Serial.printf("# frozen_export_sha256=%s nominal_input_interval_us=%u\n",
                advanced_model::EXPORT_SHA256, advanced_model::SAMPLE_INTERVAL_US);
  Serial.printf("# trials=%u samples_per_trace=%u models=%u precisions=2 inputs=%u hidden=%u\n",
                TRIALS, advanced_model::TRACE_LENGTH, advanced_model::MODEL_COUNT,
                advanced_runtime::Inputs, advanced_runtime::Hidden);
  Serial.printf("# actual_common_state_bytes=%u record_buffer_bytes=%u input_trace_bytes=%u exported_model_banks_bytes=%u heap_free=%u\n",
                static_cast<unsigned>(sizeof(state)), static_cast<unsigned>(sizeof(records)),
                static_cast<unsigned>(sizeof(advanced_model::TRACE)),
                static_cast<unsigned>(sizeof(advanced_model::MODELS)), static_cast<unsigned>(ESP.getFreeHeap()));
  Serial.println("# logical_model_parameter_bytes float32=516 int8_weight_plus_scales_float_bias=188; both banks compiled here.");
  Serial.println("# continuous input; no binary encoder or sparse input accumulation claim.");
  Serial.println("# input_prepare_us copies offline features only; raw54channel extraction/normalization excluded.");
  Serial.println("# unpaced throughput; no queue/acquisition/decision deadlines or energy measured.");
  Serial.println("# int8 storage dequantizes on the fly; arithmetic, biases, states, and readout remain float32.");
  Serial.println("# clock resolution 1us; intervals include timestamp overhead; serial excluded.");
  Serial.flush();
  for (unsigned model = 0; model < advanced_model::MODEL_COUNT; ++model)
    for (unsigned p = 0; p < 2; ++p)
      if (!replay(model, static_cast<advanced_runtime::Precision>(p), 0, false)) return;
  Serial.println("# fixture_parity=pass all_models_precisions count_exact float_tolerance=1e-4");
  Serial.println("source,run_id,model_id,model_hash,input_id,input_hash,trial,precision,step,input_prepare_us,inference_us,duration_us,hidden_spikes,final_score,decision_threshold,final_decision,reference_decision,decision_drift");
  Serial.flush();
  for (unsigned trial = 1; trial <= TRIALS; ++trial) {
    for (unsigned offset = 0; offset < advanced_model::MODEL_COUNT; ++offset) {
      const unsigned model = (offset + trial - 1) % advanced_model::MODEL_COUNT;
      for (unsigned order = 0; order < 2; ++order) {
        const unsigned p = trial % 2 ? order : 1 - order;
        if (!replay(model, static_cast<advanced_runtime::Precision>(p), trial, true)) return;
      }
    }
  }
  Serial.println("# COMPLETE hardware_advanced_replay. Recorded computation; no live gripper validation.");
}
