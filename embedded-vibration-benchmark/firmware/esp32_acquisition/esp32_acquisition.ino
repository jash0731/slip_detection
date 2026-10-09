// Experiment A only: electrical loopback, immediate decision, no sensor/filter.
// Open this file in Arduino IDE; see ../../HARDWARE.md.
#include <Arduino.h>
#include <esp_timer.h>
#include <esp_random.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/task.h>
#include <inttypes.h>

#if !defined(CONFIG_IDF_TARGET_ESP32) || CONFIG_FREERTOS_UNICORE
#error "Select a classic dual-core ESP32 board (e.g. ESP32 Dev Module)."
#endif

constexpr uint8_t STIMULUS_PIN = 25;
constexpr uint8_t INPUT_PIN = 26;
constexpr int STIMULUS_CORE = 0;
constexpr int ACQUISITION_CORE = 1;
constexpr uint32_t WIDTHS_US[] = {250, 1000, 3000, 8000, 15000};
constexpr uint32_t PERIODS_MS[] = {1, 5, 10, 20, 0}; // 0 = GPIO interrupt
constexpr uint32_t PHASE_COUNT = 20; // 5 widths x 20 phases = 100 per method
constexpr int64_t LEAD_US = 40000;
constexpr int64_t TAIL_US = 40000;
constexpr char CODE_VERSION[] = "esp32-acq-v1"; // Update when editing firmware.
constexpr char WIRING_VERSION[] = "gpio25-1k-gpio26-v1";

struct StimulusCommand {
  int64_t requestedRiseUs;
  uint32_t widthUs;
};

struct StimulusResult {
  int64_t riseUs;
  int64_t fallUs;
  int core;
};

QueueHandle_t stimulusCommands;
QueueHandle_t stimulusResults;
portMUX_TYPE decisionMux = portMUX_INITIALIZER_UNLOCKED;
// The mux protects all ISR/task accesses, including 64-bit timestamps.
volatile bool captureEnabled = false;
volatile int64_t captureStartUs = 0;
volatile int64_t captureDeadlineUs = 0;
volatile int64_t interruptDecisionUs = 0;
volatile int interruptCore = -1;
char runId[160];
bool hasRun = false;

void ARDUINO_ISR_ATTR inputRising() {
  const int64_t decisionUs = esp_timer_get_time();
  portENTER_CRITICAL_ISR(&decisionMux);
  if (captureEnabled && interruptDecisionUs == 0 && decisionUs >= captureStartUs &&
      decisionUs < captureDeadlineUs) {
    // Immediate rule: the first RISING interrupt itself is the decision.
    interruptDecisionUs = decisionUs;
    interruptCore = xPortGetCoreID();
  }
  portEXIT_CRITICAL_ISR(&decisionMux);
}

void waitUntil(int64_t targetUs) {
  for (;;) {
    const int64_t remaining = targetUs - esp_timer_get_time();
    if (remaining <= 0) return;
    // Yield during long waits; spin only near an edge. Actual edges are logged.
    if (remaining > 2 * portTICK_PERIOD_MS * 1000LL) vTaskDelay(1);
  }
}

void generateStimulus(void *) {
  StimulusCommand command;
  for (;;) {
    xQueueReceive(stimulusCommands, &command, portMAX_DELAY);
    waitUntil(command.requestedRiseUs);
    StimulusResult result;
    result.core = xPortGetCoreID();
    result.riseUs = esp_timer_get_time();
    digitalWrite(STIMULUS_PIN, HIGH);
    // Width is relative to the actual rise call, not the requested schedule.
    waitUntil(result.riseUs + command.widthUs);
    result.fallUs = esp_timer_get_time();
    digitalWrite(STIMULUS_PIN, LOW);
    xQueueSend(stimulusResults, &result, portMAX_DELAY);
  }
}

// Always quote fields and escape internal quotes (e.g. the pulse JSON).
void csvField(const char *value, bool last = false) {
  Serial.print('"');
  while (*value) {
    if (*value == '"') Serial.print('"');
    Serial.print(*value++);
  }
  Serial.print('"');
  if (last) Serial.println();
  else Serial.print(',');
}

void fatal(const char *message) {
  digitalWrite(STIMULUS_PIN, LOW);
  Serial.print("# ERROR: ");
  Serial.println(message);
  Serial.flush();
  for (;;) delay(1000);
}

void runTrial(uint32_t trialId, uint32_t intervalMs, uint32_t widthUs,
              uint32_t phaseUs) {
  if (digitalRead(INPUT_PIN) != LOW) fatal("Input must be idle LOW before a trial.");
  // An absolute epoch avoids accumulating delay/read/decision overhead.
  const int64_t epochUs = esp_timer_get_time() + 5000;
  const int64_t requestedRiseUs = epochUs + LEAD_US + phaseUs;
  const int64_t deadlineUs = requestedRiseUs + widthUs + TAIL_US;
  const int64_t periodUs = intervalMs * 1000LL;
  int64_t decisionUs = 0;
  int decisionCore = -1;
  int64_t nextPollUs = epochUs;
  int64_t maxPollLatenessUs = 0;
  uint32_t skippedPolls = 0;

  portENTER_CRITICAL(&decisionMux);
  interruptDecisionUs = 0;
  interruptCore = -1;
  captureStartUs = epochUs;
  captureDeadlineUs = deadlineUs;
  captureEnabled = (intervalMs == 0);
  portEXIT_CRITICAL(&decisionMux);
  const StimulusCommand command = {requestedRiseUs, widthUs};
  if (xQueueSend(stimulusCommands, &command, 0) != pdTRUE) {
    fatal("Stimulus command queue is full.");
  }

  // No serial output during acquisition. Keep watching until the fixed deadline,
  // even after deciding, so every method has the same trial observation window.
  while (esp_timer_get_time() < deadlineUs) {
    if (intervalMs == 0) {
      vTaskDelay(1); // ISR timestamps the decision, not this task's wakeup.
      continue;
    }
    const int64_t sampleStartUs = esp_timer_get_time();
    if (sampleStartUs < nextPollUs) continue;
    if (sampleStartUs >= deadlineUs) break;
    const int level = digitalRead(INPUT_PIN);
    const int64_t sampleDoneUs = esp_timer_get_time();
    if (level == HIGH && decisionUs == 0 && sampleDoneUs < deadlineUs) {
      decisionUs = sampleDoneUs;
      decisionCore = xPortGetCoreID();
    }
    const int64_t latenessUs = sampleStartUs - nextPollUs;
    if (latenessUs > maxPollLatenessUs) maxPollLatenessUs = latenessUs;
    nextPollUs += periodUs;
    // Skip missed ticks instead of inventing catch-up samples at one instant.
    if (nextPollUs <= sampleDoneUs) {
      const uint32_t skipped = (sampleDoneUs - nextPollUs) / periodUs + 1;
      skippedPolls += skipped;
      nextPollUs += skipped * periodUs;
    }
  }
  portENTER_CRITICAL(&decisionMux);
  captureEnabled = false;
  if (intervalMs == 0) {
    decisionUs = interruptDecisionUs;
    decisionCore = interruptCore;
  }
  portEXIT_CRITICAL(&decisionMux);

  StimulusResult stimulus;
  if (xQueueReceive(stimulusResults, &stimulus, pdMS_TO_TICKS(1000)) != pdTRUE) {
    fatal("Stimulus task did not finish; run is incomplete.");
  }
  const char *outcome = decisionUs == 0 ? "miss" :
      (decisionUs < stimulus.riseUs ? "false_alarm" : "detected");
  char trial[16], interval[16], coreConfig[80], parameters[320];
  char schedule[80], first[24], decision[24] = "", latency[24] = "", notes[320];
  snprintf(trial, sizeof(trial), "%" PRIu32, trialId);
  if (intervalMs) snprintf(interval, sizeof(interval), "%" PRIu32, intervalMs);
  else interval[0] = '\0';
  snprintf(coreConfig, sizeof(coreConfig), "generator=%d;acquisition=%d;decision=%d",
           stimulus.core, xPortGetCoreID(), decisionCore);
  snprintf(parameters, sizeof(parameters),
           "width_us=%" PRIu32 ";phase_us=%" PRIu32 ";epoch_us=%" PRId64
           ";requested_rise_us=%" PRId64 ";deadline_us=%" PRId64
           ";lead_us=%" PRId64 ";tail_us=%" PRId64 ";stimulus_priority=2",
           widthUs, phaseUs, epochUs, requestedRiseUs, deadlineUs, LEAD_US, TAIL_US);
  snprintf(schedule, sizeof(schedule), "[[%" PRId64 ",%" PRId64 "]]",
           stimulus.riseUs, stimulus.fallUs);
  snprintf(first, sizeof(first), "%" PRId64, stimulus.riseUs);
  if (decisionUs != 0) {
    snprintf(decision, sizeof(decision), "%" PRId64, decisionUs);
    snprintf(latency, sizeof(latency), "%" PRId64, decisionUs - stimulus.riseUs);
  }
  snprintf(notes, sizeof(notes),
           "source=hardware;stimulus=gpio_loopback;cpu_mhz=%" PRIu32
           ";acquisition_priority=%u;max_poll_lateness_us=%" PRId64
           ";skipped_polls=%" PRIu32 ";stimulus_after_deadline=%d;pre_stimulus_decision=%d",
           getCpuFrequencyMhz(), unsigned(uxTaskPriorityGet(nullptr)),
           maxPollLatenessUs, skippedPolls, stimulus.fallUs >= deadlineUs,
           decisionUs != 0 && decisionUs < stimulus.riseUs);
  csvField(runId);
  csvField(trial);
  csvField("target");
  csvField(intervalMs ? "polling" : "interrupt");
  csvField("immediate");
  csvField(interval);
  csvField(coreConfig);
  csvField("none;wifi=off;bluetooth=off");
  csvField(parameters);
  csvField(schedule);
  csvField(first);
  csvField(decision);
  csvField(latency);
  csvField(outcome);
  csvField(notes, true);
  Serial.flush(); // Drain the row before the next trial starts.
  delay(10); // Let idle tasks/watchdogs run between bounded observation windows.
}

void setup() {
  Serial.begin(115200);
  pinMode(STIMULUS_PIN, OUTPUT);
  digitalWrite(STIMULUS_PIN, LOW);
  pinMode(INPUT_PIN, INPUT_PULLDOWN);
  delay(1500);
  if (xPortGetCoreID() != ACQUISITION_CORE) {
    fatal("Set Arduino Runs On = Core 1 in the board menu.");
  }
  // No Wi-Fi/Bluetooth APIs are started by this sketch.
  // A static HIGH/LOW preflight catches a missing wire without faking success.
  digitalWrite(STIMULUS_PIN, HIGH);
  delay(2);
  const bool highOk = digitalRead(INPUT_PIN) == HIGH;
  digitalWrite(STIMULUS_PIN, LOW);
  delay(2);
  if (!highOk || digitalRead(INPUT_PIN) != LOW) {
    fatal("Loopback check failed. Connect GPIO25 through 1k ohm to GPIO26.");
  }
  stimulusCommands = xQueueCreate(1, sizeof(StimulusCommand));
  stimulusResults = xQueueCreate(1, sizeof(StimulusResult));
  if (!stimulusCommands || !stimulusResults) fatal("Queue allocation failed.");
  if (xTaskCreatePinnedToCore(generateStimulus, "stimulus", 4096, nullptr, 2,
                              nullptr, STIMULUS_CORE) != pdPASS) {
    fatal("Stimulus task allocation failed.");
  }
  snprintf(runId, sizeof(runId), "hardware-%s-%s-%012" PRIx64 "-%08" PRIx32,
           CODE_VERSION, WIRING_VERSION, ESP.getEfuseMac(), esp_random());
  Serial.println("# ESP32 experiment A; HARDWARE electrical loopback only");
  Serial.print("# run_id="); Serial.println(runId);
  Serial.println("# Send s to start once; reset to run again.");
}

void loop() {
  if (hasRun || !Serial.available() || Serial.read() != 's') {
    delay(10);
    return;
  }
  hasRun = true;
  Serial.println("run_id,trial_id,trial_kind,acquisition,detector,poll_interval_ms,core_config,load,parameters,pulse_schedule_us,first_pulse_us,decision_us,latency_us,outcome,notes");
  Serial.flush();
  uint32_t trialId = 1;
  for (uint32_t intervalMs : PERIODS_MS) {
    if (intervalMs == 0) attachInterrupt(INPUT_PIN, inputRising, RISING);
    for (uint32_t widthUs : WIDTHS_US) {
      for (uint32_t phase = 0; phase < PHASE_COUNT; ++phase) {
        // Stratify across 20 ms with sub-ms offsets, identical in every method.
        const uint32_t phaseUs = phase * 1000 + (phase * 347 + 137) % 1000;
        runTrial(trialId++, intervalMs, widthUs, phaseUs);
      }
    }
    if (intervalMs == 0) detachInterrupt(INPUT_PIN);
  }
  Serial.println("# COMPLETE: 500 hardware target trials; reset for another run.");
}
