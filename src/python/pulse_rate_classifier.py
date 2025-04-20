import numpy as np
import pandas as pd
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

class PulseRateClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def preprocess_signal(self, raw_signal, sampling_rate):
        # Apply bandpass filter to remove noise
        nyquist = sampling_rate / 2
        low = 0.5 / nyquist  # 0.5 Hz cutoff for low frequency
        high = 8.0 / nyquist  # 8.0 Hz cutoff for high frequency
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, raw_signal)
        return filtered_signal
    
    def extract_features(self, filtered_signal, sampling_rate):
        # Time domain features
        mean_hr = np.mean(filtered_signal)
        std_hr = np.std(filtered_signal)
        rms = np.sqrt(np.mean(filtered_signal**2))
        
        # Heart rate variability features
        peaks, _ = signal.find_peaks(filtered_signal, distance=sampling_rate//2)
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / sampling_rate
            hrv_std = np.std(rr_intervals)
        else:
            hrv_std = 0
        
        features = {
            'mean_hr': mean_hr,
            'std_hr': std_hr,
            'rms': rms,
            'hrv_std': hrv_std
        }
        
        return features
    
    def train(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        # Train model
        self.model.fit(X_scaled, y)
    
    def predict(self, X):
        # Scale features
        X_scaled = self.scaler.transform(X)
        # Make predictions
        return self.model.predict(X_scaled)
    
    def evaluate(self, X_test, y_test):
        # Make predictions
        y_pred = self.predict(X_test)
        # Print classification report
        print(classification_report(y_test, y_pred))

# Example usage
def main():
    # This is a placeholder for actual data loading
    # In real application, you would load your pulse rate data here
    # Example: data = load_pulse_rate_data('path_to_data')
    
    # For demonstration, let's create some synthetic data
    np.random.seed(42)
    sampling_rate = 100  # Hz
    duration = 60  # seconds
    t = np.linspace(0, duration, duration * sampling_rate)
    
    # Generate synthetic normal and abnormal heart signals
    normal_signals = []
    abnormal_signals = []
    
    for _ in range(50):  # Generate 50 samples of each class
        # Normal heart rate: clean sine wave with frequency around 1-1.5 Hz
        normal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t))
        normal_signals.append(normal)
        
        # Abnormal heart rate: irregular frequency and more noise
        abnormal = np.sin(2 * np.pi * 1.2 * t) * np.sin(2 * np.pi * 0.1 * t) + 0.3 * np.random.randn(len(t))
        abnormal_signals.append(abnormal)
    
    # Create classifier instance
    classifier = PulseRateClassifier()
    
    # Process signals and extract features
    features = []
    labels = []
    
    for signal_data in normal_signals:
        filtered = classifier.preprocess_signal(signal_data, sampling_rate)
        feature_dict = classifier.extract_features(filtered, sampling_rate)
        features.append(list(feature_dict.values()))
        labels.append(0)  # 0 for normal
    
    for signal_data in abnormal_signals:
        filtered = classifier.preprocess_signal(signal_data, sampling_rate)
        feature_dict = classifier.extract_features(filtered, sampling_rate)
        features.append(list(feature_dict.values()))
        labels.append(1)  # 1 for abnormal
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate
    classifier.train(X_train, y_train)
    print("Model Evaluation:")
    classifier.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()