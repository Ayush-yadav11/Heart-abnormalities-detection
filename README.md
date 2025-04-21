# Heart Abnormalities Detection System

This project implements an advanced real-time heart monitoring system using the MAX30102 sensor, ESP32 microcontroller, and machine learning for comprehensive heart abnormality detection. The system provides continuous monitoring of vital signs including pulse rate and blood oxygen saturation (SpO2), with real-time alerts and visualization through an interactive Node-RED dashboard.

## Key Features

- Real-time pulse rate monitoring with abnormality detection
- Continuous SpO2 level tracking and analysis
- Machine learning-based heart pattern classification
- Interactive dashboard with real-time visualization
- Automated alerts for abnormal heart conditions
- Historical data tracking and analysis

## System Architecture

The system consists of three main components:

1. **Data Collection (Arduino)**
   - MAX30102 sensor for pulse rate measurements
   - ESP32 board for sensor interfacing and data acquisition
   - Real-time data streaming via serial communication

2. **Data Processing (Python)**
   - Real-time data processing and feature extraction
   - Machine learning model for abnormality detection
   - Integration with Node-RED for visualization

3. **Visualization (Node-RED)**
   - Real-time dashboard for pulse rate monitoring
   - Visual alerts for abnormal heart rate patterns
   - Historical data visualization and analysis

## Hardware Requirements

- ESP32 Development Board
- MAX30102 Pulse Oximeter and Heart Rate Sensor
- USB cable for ESP32
- Jumper wires

## Software Dependencies

1. **ESP32 Setup**
   - Arduino IDE with ESP32 board support
   - MAX30102 sensor library

2. **Python Environment**
   - Python 3.x
   - Required packages: numpy, pandas, scikit-learn, pyserial
   - Install dependencies: `pip install -r requirements.txt`

3. **Node-RED**
   - Node.js and npm
   - Node-RED installation
   - Required nodes: dashboard, serial

## Project Structure

- `src/arduino/` - Arduino sketches and sensor interface
  - `max30102_sensor.ino`: Main Arduino code for sensor data collection

- `src/python/` - Data processing and machine learning
  - `max30102_reader.py`: Interface for reading sensor data
  - `pulse_rate_classifier.py`: ML model for heart rate classification
  - `real_time_monitor.py`: Main monitoring script
  - `train_model.py`: Model training script
  - `evaluate_model.py`: Model evaluation and visualization

- `src/node-red/` - Dashboard and visualization
  - `flows.json`: Node-RED flow configurations
  - `dashboard.css`: Custom dashboard styling

- `src/data/` - Datasets and trained models
  - `pulse_rate_dataset.csv`: Training dataset
  - `pulse_rate_model.joblib`: Trained ML model

## Setup Instructions

1. **Hardware Setup**
   - Connect MAX30102 sensor to ESP32 using appropriate pins
   - Power the ESP32 via USB

2. **ESP32 Programming**
   - Upload the Arduino sketch from `src/arduino/max30102_sensor.ino`
   - Configure serial communication settings

3. **Python Environment**
   - Install required Python packages
   - Run `train_model.py` to train the ML model
   - Start the monitoring script: `python real_time_monitor.py`

4. **Node-RED Setup**
   - Import the flow configuration from `src/node-red/flows.json`
   - Configure the serial node to match ESP32 port
   - Deploy the flow and access the dashboard

## Usage

1. Start the Node-RED server and access the dashboard
2. Run the Python monitoring script
3. The dashboard will display:
   - Real-time pulse rate readings
   - Abnormality detection alerts
   - Historical data visualization

## Real-time Monitoring and Visualization

The system provides comprehensive real-time monitoring of vital signs with the following features:

1. **Pulse Rate Monitoring**
   - Real-time visualization of heart rate in Beats Per Minute (BPM)
   - Abnormality detection thresholds:
     - Bradycardia: Heart rate below 60 BPM
     - Tachycardia: Heart rate above 100 BPM
   - Visual alerts for abnormal heart patterns

2. **SpO2 Level Monitoring**
   - Continuous tracking of blood oxygen saturation levels
   - Normal range: 80-100%
   - Real-time graphical representation

3. **Data Visualization Features**
   - Dual-panel display showing both pulse rate and SpO2
   - Historical data view with 200-point window
   - Grid-based layout for precise readings
   - Color-coded threshold lines for quick abnormality detection

## Data Format

The system processes data in a simplified CSV format with the following structure:

```csv
timestamp,pulse_rate,spo2
2024-01-01 10:00:00,75,98
2024-01-01 10:00:01,72,97
```

## Model Performance and Thresholds

The system implements clinical-standard thresholds for abnormality detection:

- **Bradycardia Detection**: Alerts when pulse rate < 60 BPM
- **Tachycardia Detection**: Alerts when pulse rate > 100 BPM
- **SpO2 Monitoring**: Tracks oxygen saturation with normal range 95-100%

The machine learning model achieves robust performance metrics:

- **Accuracy**: >95% classification accuracy on test data
- **Precision**: >90% true positive rate for abnormality detection
- **Recall**: >92% detection rate of actual abnormalities
- **Response Time**: <500ms for real-time analysis

## Documentation

Detailed documentation, including system diagrams and API references, can be found in the `src/docs/` directory.
