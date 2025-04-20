from max30102_reader import MAX30102Reader
from pulse_rate_classifier import PulseRateClassifier
import numpy as np
from time import sleep

def train_classifier_with_synthetic_data():
    # Generate synthetic training data similar to original implementation
    np.random.seed(42)
    sampling_rate = 100  # Hz
    duration = 60  # seconds
    t = np.linspace(0, duration, duration * sampling_rate)
    
    normal_signals = []
    abnormal_signals = []
    
    for _ in range(50):
        normal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t))
        normal_signals.append(normal)
        
        abnormal = np.sin(2 * np.pi * 1.2 * t) * np.sin(2 * np.pi * 0.1 * t) + 0.3 * np.random.randn(len(t))
        abnormal_signals.append(abnormal)
    
    classifier = PulseRateClassifier()
    features = []
    labels = []
    
    # Process normal signals
    for signal_data in normal_signals:
        filtered = classifier.preprocess_signal(signal_data, sampling_rate)
        feature_dict = classifier.extract_features(filtered, sampling_rate)
        features.append(list(feature_dict.values()))
        labels.append(0)
    
    # Process abnormal signals
    for signal_data in abnormal_signals:
        filtered = classifier.preprocess_signal(signal_data, sampling_rate)
        feature_dict = classifier.extract_features(filtered, sampling_rate)
        features.append(list(feature_dict.values()))
        labels.append(1)
    
    # Train the classifier
    classifier.train(np.array(features), np.array(labels))
    return classifier

def main():
    # Initialize sensor reader
    sensor = MAX30102Reader()
    
    # Train classifier with synthetic data
    classifier = train_classifier_with_synthetic_data()
    
    print("Starting real-time monitoring...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            # Get 1 second of data
            signal_window = sensor.get_signal_window(duration=1)
            
            if len(signal_window) == sensor.sampling_rate:
                # Preprocess the signal
                filtered_signal = classifier.preprocess_signal(signal_window, sensor.sampling_rate)
                
                # Extract features
                features = classifier.extract_features(filtered_signal, sensor.sampling_rate)
                
                # Make prediction
                prediction = classifier.predict([list(features.values())])[0]
                
                # Print result
                status = "Normal" if prediction == 0 else "Abnormal"
                print(f"Heart Rate Status: {status}")
            
            sleep(0.1)  # Small delay to prevent CPU overload
            
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
    finally:
        sensor.close()

if __name__ == "__main__":
    main()