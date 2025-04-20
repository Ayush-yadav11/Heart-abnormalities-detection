# Python Scripts Directory

This directory contains Python scripts for data processing, model training, and real-time monitoring.

## Files

- `max30102_reader.py` - Interface with MAX30102 sensor
  - Handles serial communication with Arduino
  - Processes raw sensor data
  - Implements data validation and error handling
  - Provides clean data interface for other components

- `pulse_rate_classifier.py` - Machine learning model for pulse rate classification
  - Implements the pulse rate classification algorithm
  - Provides methods for data preprocessing
  - Includes feature extraction functions
  - Handles model prediction and output formatting

- `train_model.py` - Script for training the pulse rate model
  - Loads and preprocesses training data
  - Implements model training pipeline
  - Performs model validation
  - Saves trained model to joblib file

- `evaluate_model.py` - Model evaluation and metrics generation
  - Calculates model performance metrics
  - Generates evaluation visualizations
  - Performs cross-validation
  - Outputs detailed performance reports

- `real_time_monitor.py` - Real-time pulse rate monitoring
  - Integrates sensor reading and model prediction
  - Implements real-time data visualization
  - Handles data logging and storage
  - Provides Node-RED integration interface
  