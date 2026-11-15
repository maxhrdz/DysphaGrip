#include <Wire.h>
#include <Arduino_RouterBridge.h>

const uint8_t MPU_ADDR = 0x68;
const uint32_t SAMPLE_PERIOD_MS = 40;  // Objetivo inicial: 25 lecturas/s.

String frame;
uint32_t lastStart;
uint32_t sequence = 0;
bool sensorAwake = false;

int16_t read16() {
  uint16_t hi = Wire.read();
  uint16_t lo = Wire.read();
  return static_cast<int16_t>((hi << 8) | lo);
}

bool wakeSensor() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  return Wire.endTransmission() == 0;
}

void setup() {
  Monitor.begin();
  Wire.begin();
  frame.reserve(128);
  sensorAwake = wakeSensor();
  delay(2000);
  Monitor.println("DysphaGrip motion test started");
  lastStart = millis() - SAMPLE_PERIOD_MS;
}

void loop() {
  const uint32_t elapsed = static_cast<uint32_t>(millis()) - lastStart;
  if (elapsed < SAMPLE_PERIOD_MS) {
    delay(SAMPLE_PERIOD_MS - elapsed);
  }
  lastStart = millis();

  if (!sensorAwake) {
    sensorAwake = wakeSensor();
    if (!sensorAwake) {
      Monitor.println("Sensor not responding - check wires");
      delay(500);
      return;
    }
    delay(100);
  }

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0 ||
      Wire.requestFrom(MPU_ADDR, (size_t)14) != 14) {
    sensorAwake = false;
    Monitor.println("I2C communication error");
    delay(500);
    return;
  }

  const uint32_t sampleTime = millis();
  const float ax = read16() / 16384.0f;
  const float ay = read16() / 16384.0f;
  const float az = read16() / 16384.0f;
  read16();  // Temperatura.
  const float gx = read16() / 131.0f;
  const float gy = read16() / 131.0f;
  const float gz = read16() / 131.0f;

  frame = "Tilt X: ";
  frame += String(ax, 2);
  frame += " Y: ";
  frame += String(ay, 2);
  frame += " Z: ";
  frame += String(az, 2);
  frame += " | Turn X: ";
  frame += String(gx, 1);
  frame += " Y: ";
  frame += String(gy, 1);
  frame += " Z: ";
  frame += String(gz, 1);
  frame += " seq=";
  frame += String(static_cast<unsigned long>(sequence++));
  frame += " t_ms=";
  frame += String(static_cast<unsigned long>(sampleTime));
  frame += '\n';
  Monitor.write(reinterpret_cast<const uint8_t*>(frame.c_str()), frame.length());
}
