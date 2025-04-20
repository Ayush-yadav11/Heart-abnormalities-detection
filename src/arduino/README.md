# Arduino Code Directory

This directory contains the Arduino sketches for the MAX30102 pulse rate sensor.

## Files

- `max30102_sensor.ino` - Main Arduino sketch for reading sensor data from the MAX30102 sensor. This sketch:
  - Initializes and configures the MAX30102 sensor
  - Reads raw IR and Red LED values
  - Calculates heart rate and SpO2 values
  - Sends data via Serial communication to the Python interface
  - Handles sensor error detection and reporting
  