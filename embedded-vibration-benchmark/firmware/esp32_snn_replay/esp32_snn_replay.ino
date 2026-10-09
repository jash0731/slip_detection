// Sensor-free compute replay for robotic-gripper slip research.
// Exported recorded/synthetic input provenance is printed; no live slip sensing.
#include <Arduino.h>
#include <esp_timer.h>
#include <esp_system.h>
#include <esp_random.h>
#include <inttypes.h>
#include "generated_model.h"
#include "snn_runtime.h"

using Network = gripper_runtime::Network<gripper_model::INPUTS, gripper_model::HIDDEN>;
Network network;
constexpr unsigned TRIALS = 10;
constexpr unsigned ACTIVITY_STEPS = 10000;
constexpr float ACTIVITY_AMPLITUDES[] = {0.0f, 0.2f, 0.4f, 0.7f, 1.2f};
constexpr char CODE_ID[] = "gripper-snn-replay-v1";
struct StepRecord {
  uint32_t activeInputs;
  uint32_t hiddenSpikes;
  int64_t encodeUs;
  int64_t inferenceUs;
  int64_t startLatenessUs;
  bool deadlineMiss;
};
StepRecord records[gripper_model::TRACE_LENGTH];
char runId[96];
bool finished = false;
bool paced = false;
struct ActivityResult {
  int64_t durationUs;
  double checksum;
  float voltage[gripper_model::HIDDEN];
  uint32_t counts[gripper_model::HIDDEN];
};
ActivityResult activityResults[2];
volatile double activityChecksumSink = 0;

void activityBatch(float amplitude, gripper_runtime::Mode mode, unsigned slot) {
  network.reset();
  uint8_t events[gripper_model::INPUTS];
  Network::encode(amplitude, gripper_model::INPUT_THRESHOLDS, events);
  const int64_t start = esp_timer_get_time();
  for (unsigned tick = 0; tick < ACTIVITY_STEPS; ++tick)
    network.step(events, gripper_model::WEIGHTS, gripper_model::LEAK,
                 gripper_model::THRESHOLD, gripper_model::RESET, mode);
  const int64_t end = esp_timer_get_time();
  ActivityResult &result = activityResults[slot];
  result.durationUs = end - start;
  result.checksum = 0;
  for (unsigned h = 0; h < gripper_model::HIDDEN; ++h) {
    result.voltage[h] = network.voltage[h];
    result.counts[h] = network.counts[h];
    result.checksum += network.counts[h] + static_cast<double>(network.voltage[h]);
  }
  activityChecksumSink += result.checksum; // Observable result, outside timing.
}

bool activityStatesMatch() {
  for (unsigned h = 0; h < gripper_model::HIDDEN; ++h)
    if (activityResults[0].counts[h] != activityResults[1].counts[h] ||
        activityResults[0].voltage[h] != activityResults[1].voltage[h]) return false;
  return true;
}

void activitySweep() {
  Serial.println("# Controlled synthetic constant-input workload; public-trained readout NOT evaluated.");
  Serial.printf("# source=hardware_activity_sweep trials=%u steps=%u conditions=5 inputs=%u hidden=%u\n",
                TRIALS, ACTIVITY_STEPS, gripper_model::INPUTS, gripper_model::HIDDEN);
  Serial.println("# timing=hidden_updates_only encoder_reset_readout_serial_excluded; timer overhead included.");
  Serial.println("# input_contributions=source_level_weight_terms not hardware_instruction_counts.");
  Serial.println("source,run_id,model_id,trial,mode,active_inputs,input_channels,steps,duration_us,input_contributions,hidden_updates,checksum");
  Serial.flush();
  for (float amplitude : ACTIVITY_AMPLITUDES) {
    uint8_t events[gripper_model::INPUTS];
    const unsigned active = Network::encode(amplitude, gripper_model::INPUT_THRESHOLDS, events);
    activityBatch(amplitude, gripper_runtime::Mode::Dense, 0);
    activityBatch(amplitude, gripper_runtime::Mode::Sparse, 1);
    if (!activityStatesMatch()) {
      Serial.println("# ERROR activity warm-up parity mismatch; incomplete run."); return;
    }
    for (unsigned trial = 1; trial <= TRIALS; ++trial) {
      const unsigned first = trial % 2 ? 0 : 1;
      const unsigned second = 1 - first;
      activityBatch(amplitude, first ? gripper_runtime::Mode::Sparse : gripper_runtime::Mode::Dense, first);
      activityBatch(amplitude, second ? gripper_runtime::Mode::Sparse : gripper_runtime::Mode::Dense, second);
      if (!activityStatesMatch()) {
        Serial.println("# ERROR activity trial parity mismatch; incomplete run."); return;
      }
      for (unsigned order = 0; order < 2; ++order) {
        const unsigned slot = order ? second : first;
        const ActivityResult &r = activityResults[slot];
        const unsigned contributions = (slot ? active : gripper_model::INPUTS) *
                                        gripper_model::HIDDEN * ACTIVITY_STEPS;
        Serial.printf("hardware_activity_sweep,%s,%s,%u,%s,%u,%u,%u,%" PRId64 ",%u,%u,%.17g\n",
                      runId, gripper_model::MODEL_ID, trial, slot ? "sparse" : "dense",
                      active, gripper_model::INPUTS, ACTIVITY_STEPS, r.durationUs,
                      contributions, gripper_model::HIDDEN * ACTIVITY_STEPS, r.checksum);
      }
      Serial.flush();
      delay(5); // Yield between batches, never inside the measured loop.
    }
  }
  Serial.println("# activity_state_parity=pass all_conditions_trials exact_spike_counts_and_voltages");
  Serial.println("# COMPLETE hardware_activity_sweep. No live slip accuracy or energy measurements.");
}

bool fixtureMatches() {
  for (unsigned h = 0; h < gripper_model::HIDDEN; ++h) {
    if (network.counts[h] != static_cast<uint32_t>(gripper_model::EXPECTED_COUNTS[h]) ||
        fabsf(network.voltage[h] - gripper_model::EXPECTED_VOLTAGE[h]) > 1e-5f) return false;
  }
  return fabsf(network.score(gripper_model::READOUT, gripper_model::BIAS) -
               gripper_model::EXPECTED_SCORE) <= 1e-5f;
}

void waitUntil(int64_t target) {
  while (esp_timer_get_time() < target) {
    if (target - esp_timer_get_time() > 2000) delay(1);
  }
}

void replay(gripper_runtime::Mode mode, unsigned trial, bool printRows) {
  network.reset();
  const int64_t traceStart = esp_timer_get_time();
  unsigned deadlineMisses = 0;
  for (unsigned step = 0; step < gripper_model::TRACE_LENGTH; ++step) {
    uint8_t events[gripper_model::INPUTS];
    const int64_t scheduled = traceStart + static_cast<int64_t>(step) * gripper_model::SAMPLE_INTERVAL_US;
    if (paced) waitUntil(scheduled);
    const int64_t start = esp_timer_get_time();
    const unsigned active = Network::encode(gripper_model::TRACE[step],
                                           gripper_model::INPUT_THRESHOLDS, events);
    const int64_t encoded = esp_timer_get_time();
    const unsigned spikes = network.step(events, gripper_model::WEIGHTS,
                                        gripper_model::LEAK, gripper_model::THRESHOLD,
                                        gripper_model::RESET, mode);
    const int64_t inferred = esp_timer_get_time();
    const bool miss = paced && inferred > scheduled + gripper_model::SAMPLE_INTERVAL_US;
    deadlineMisses += miss;
    records[step] = {active, spikes, encoded - start, inferred - encoded,
                     paced ? start - scheduled : 0, miss};
  }
  const int64_t readoutStart = esp_timer_get_time();
  const float score = network.score(gripper_model::READOUT, gripper_model::BIAS);
  const int64_t end = esp_timer_get_time();
  if (!printRows) return;
  const char *modeName = mode == gripper_runtime::Mode::Dense ? "dense" : "sparse";
  Serial.printf("# trace trial=%u mode=%s elapsed_us=%" PRId64
                " readout_us=%" PRId64 " samples=%u score=%.9g decision=%d deadline_misses=%u\n",
                trial, modeName, end - traceStart, end - readoutStart,
                static_cast<unsigned>(network.samples), static_cast<double>(score), score >= 0.0f, deadlineMisses);
  for (unsigned step = 0; step < gripper_model::TRACE_LENGTH; ++step) {
    const StepRecord &r = records[step];
    // Final recording score is repeated for convenient aggregation. It is not
    // a per-tick classifier output or an incipient-slip timestamp.
    Serial.printf("hardware_replay,%s,%s,%s,%u,%s,%s,%u,%u,%u,%" PRId64
                  ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%d,%.9g,%d\n",
                  runId, gripper_model::MODEL_ID, gripper_model::INPUT_ID, trial,
                  modeName, paced ? "paced" : "throughput", step,
                  static_cast<unsigned>(r.activeInputs), static_cast<unsigned>(r.hiddenSpikes),
                  r.encodeUs, r.inferenceUs, r.encodeUs + r.inferenceUs,
                  r.startLatenessUs, r.deadlineMiss,
                  static_cast<double>(score), score >= 0.0f);
  }
  Serial.flush();
  delay(5);
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  snprintf(runId, sizeof(runId), "%s-%08" PRIx32, CODE_ID, esp_random());
  Serial.println("# Hardware compute replay only; inspect exported input/model provenance.");
  Serial.println("# No live sensor acquisition, GPIO stimulus, or gripper control.");
  Serial.println("# Press s throughput, p paced replay, or a activity sweep, then Enter; reset to rerun.");
}

void loop() {
  if (finished || !Serial.available()) { delay(10); return; }
  const char command = Serial.read();
  if (command != 's' && command != 'p' && command != 'a') return;
  paced = command == 'p';
  finished = true;
  Serial.printf("# code=%s chip=%s cpu_mhz=%u sdk=%s run_id=%s\n", CODE_ID,
                ESP.getChipModel(), static_cast<unsigned>(ESP.getCpuFreqMHz()), ESP.getSdkVersion(), runId);
  if (command == 'a') { activitySweep(); return; }
  Serial.printf("# model=%s input=%s source=hardware_replay input_origin=%s\n",
                gripper_model::MODEL_ID, gripper_model::INPUT_ID, gripper_model::DATA_ORIGIN);
  Serial.printf("# input_description=%s\n", gripper_model::INPUT_DESCRIPTION);
  Serial.printf("# trials=%u samples_per_trace=%u inputs=%u hidden=%u\n",
                TRIALS, gripper_model::TRACE_LENGTH, gripper_model::INPUTS, gripper_model::HIDDEN);
  Serial.printf("# model_state_bytes=%u trace_buffer_bytes=%u replay_record_bytes=%u"
                " heap_free=%u heap_min_free=%u\n",
                static_cast<unsigned>(sizeof(network)),
                static_cast<unsigned>(sizeof(gripper_model::TRACE)),
                static_cast<unsigned>(sizeof(records)), static_cast<unsigned>(ESP.getFreeHeap()),
                static_cast<unsigned>(ESP.getMinFreeHeap()));
  Serial.printf("# pacing=%s nominal_interval_us=%u; throughput mode has no deadlines.\n",
                paced ? "paced" : "throughput", gripper_model::SAMPLE_INTERVAL_US);
  Serial.println("# Paced deadlines end each nominal interval; all overdue steps are processed, no drops.");
  Serial.println("# esp_timer microsecond clock; short steps may report zero; timestamp overhead included.");
  Serial.println("# elapsed trace includes timestamp calls/buffer writes/readout; serial output excluded.");
  Serial.flush();
  replay(gripper_runtime::Mode::Dense, 0, false);
  if (!fixtureMatches()) { Serial.println("# ERROR dense fixture mismatch; no benchmark emitted."); return; }
  replay(gripper_runtime::Mode::Sparse, 0, false); // Warm-up output suppressed.
  if (!fixtureMatches()) { Serial.println("# ERROR sparse fixture mismatch; no benchmark emitted."); return; }
  Serial.println("# fixture_parity=pass tolerance=1e-5 both_modes; exact hidden spike counts.");
  Serial.println("source,run_id,model_id,input_id,trial,mode,pacing,step,active_inputs,hidden_spikes,encode_us,inference_us,duration_us,start_lateness_us,deadline_miss,final_score,final_decision");
  Serial.flush();
  for (unsigned trial = 1; trial <= TRIALS; ++trial) {
    // Alternate ordering to avoid always giving one mode the first position.
    const auto first = trial % 2 ? gripper_runtime::Mode::Dense : gripper_runtime::Mode::Sparse;
    const auto second = trial % 2 ? gripper_runtime::Mode::Sparse : gripper_runtime::Mode::Dense;
    replay(first, trial, true);
    replay(second, trial, true);
  }
  Serial.println("# COMPLETE. Replay compute only; not live or incipient-slip validation.");
}
