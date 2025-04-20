# Data Directory

This directory contains dataset files and trained models for the pulse rate monitoring project.

## Files

- `pulse_rate_dataset.csv` - Training dataset with labeled pulse rate data
  - Contains timestamped pulse rate measurements
  - Includes features extracted from raw sensor data
  - Used for training the machine learning model

- `ffffinal.csv` - Real-time pulse rate measurements
  - Stores continuous monitoring data
  - Includes timestamp, heart rate, and SpO2 values
  - Used for analysis and visualization

- `pulse_rate_model.joblib` - Trained machine learning model
  - Serialized scikit-learn model
  - Used for real-time pulse rate classification
  - Compatible with the pulse_rate_classifier.py script
  