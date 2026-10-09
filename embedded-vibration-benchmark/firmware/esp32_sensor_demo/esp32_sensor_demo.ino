// Live SW-420 digital-output demonstration; not a trained slip detector.
// Sensor VCC -> 3V3, GND -> GND, DO -> GPIO27.
// GPIO4 -> 1k resistor -> LED anode; LED cathode -> GND.
const int SENSOR_PIN = 27;
const int LED_PIN = 4;
int idleLevel;
int previousLevel;
unsigned long ledUntil = 0;
unsigned long lastStatus = 0;

void setup() {
  Serial.begin(115200);
  pinMode(SENSOR_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH); // Brief startup check for LED wiring.
  delay(500);
  digitalWrite(LED_PIN, LOW);
  Serial.println("Keep sensor still: calibrating idle level for 2 seconds.");
  unsigned int highCount = 0;
  for (int i = 0; i < 2000; ++i) {
    highCount += digitalRead(SENSOR_PIN) == HIGH;
    delay(1);
  }
  idleLevel = highCount >= 1000 ? HIGH : LOW;
  previousLevel = digitalRead(SENSOR_PIN);
  Serial.println("LIVE VIBRATION DEMO: tap sensor gently. Send c to recalibrate.");
  Serial.println("millis,raw_level,idle_level,led_on");
}

void loop() {
  unsigned long now = millis();
  int level = digitalRead(SENSOR_PIN);
  if (Serial.available() && Serial.read() == 'c') {
    idleLevel = level;
    Serial.println("# idle level recalibrated; keep sensor still when sending c");
  }
  if (level != idleLevel) ledUntil = now + 150;
  bool ledOn = (long)(ledUntil - now) > 0;
  digitalWrite(LED_PIN, ledOn ? HIGH : LOW);
  if (level != previousLevel || now - lastStatus >= 250) {
    Serial.printf("%lu,%d,%d,%d\n", now, level, idleLevel, ledOn);
    previousLevel = level;
    lastStatus = now;
  }
}
