# Live vibration and LED demo

Open `esp32_sensor_demo.ino` in Arduino IDE. Select ESP32 Dev Module and the USB serial port, then upload. Open Serial Monitor at 115200 baud.

With USB disconnected, wire the sensor VCC to 3V3, sensor GND to ESP32 GND, and sensor DO to GPIO27. Connect GPIO4 through a 1 kΩ resistor to the LED long leg; connect its short leg to GND. A breadboard may be used only for shared power rails. Never connect DO to a power rail.

The LED flashes for 500 ms at startup. Keep the sensor still for the following two-second idle calibration, then tap it gently. A departure from its idle digital level lights the LED for at least 150 ms. If adjusting the module potentiometer changes the idle level, keep it still and send `c` to recalibrate.

Serial output reports actual hardware digital input and LED state. This is a sensor wiring demonstration, not a trained robotic-gripper slip detector, and its CSV is separate from the benchmark schema in EXPERIMENT.md. The previously captured replay results remain in hardware_results.
