#include <Wire.h>
#include "MAX30105.h"

#define I2C_SDA 21
#define I2C_SCL 22

MAX30105 particleSensor;

void setup() {
  Serial.begin(115200);
  
  Wire.begin(I2C_SDA, I2C_SCL);
  
  
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30105 was not found. Please check wiring/power.");
    while (1);
  }
  
 
  particleSensor.setup();
  particleSensor.setPulseAmplitudeRed(0x0A);
  particleSensor.setPulseAmplitudeGreen(0);
  particleSensor.setPulseAmplitudeIR(0x0A);
  particleSensor.setSampleRate(100);
}

void loop() {
  
  uint32_t ir = particleSensor.getIR();
  uint32_t red = particleSensor.getRed();
  
  
  Serial.print(ir);
  Serial.print(",");
  Serial.println(red);
  
 
  delay(10);
}