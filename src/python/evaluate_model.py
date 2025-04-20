import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from pulse_rate_classifier import PulseRateClassifier
import seaborn as sns

def load_real_data(file_path):
    """Load the real sensor data from CSV"""
    print(f"Loading real sensor data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded with {len(df)} samples")
    print(f"Dataset columns: {df.columns.tolist()}")
    return df

def preprocess_real_data(df):
    """Preprocess the real sensor data for evaluation"""
    # Filter out extreme values (0 and 4095)
    filtered_df = df[(df['Pulse'] != 0) & (df['Pulse'] != 4095)].copy()
    print(f"After filtering extreme values, {len(filtered_df)} samples remain")
    
    # Define normal and abnormal pulse ranges
    # Normal pulse range (in our sensor units): 1500-3000
    
    # Label data
    filtered_df['label'] = ((filtered_df['Pulse'] >= 1500) & 
                           (filtered_df['Pulse'] <= 3000)).astype(int)
    filtered_df['label'] = 1 - filtered_df['label']  # Invert to make normal=0, abnormal=1
    
    print(f"Class distribution:\n{filtered_df['label'].value_counts()}")
    return filtered_df

def extract_features(df):
    """Extract features from the real sensor data"""
    features = []
    labels = []
    
    # Group by label
    for label, group in df.groupby('label'):
        # Basic statistical features
        pulse_mean = group['Pulse'].mean()
        pulse_std = group['Pulse'].std()
        pulse_range = group['Pulse'].max() - group['Pulse'].min()
        
        # Create feature vector
        feature_vector = [
            pulse_mean, pulse_std, pulse_range
        ]
        
        features.append(feature_vector)
        labels.append(label)
    
    # Create sliding windows for more samples
    window_size = 10
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i+window_size]
        
        # Only use windows with consistent labels
        if window['label'].nunique() == 1:
            label = window['label'].iloc[0]
            
            # Extract features from window
            pulse_mean = window['Pulse'].mean()
            pulse_std = window['Pulse'].std()
            pulse_range = window['Pulse'].max() - window['Pulse'].min()
            
            # Create feature vector
            feature_vector = [
                pulse_mean, pulse_std, pulse_range
            ]
            
            features.append(feature_vector)
            labels.append(label)
    
    return np.array(features), np.array(labels)

def evaluate_model_with_cross_validation(X, y, n_splits=5):
    """Evaluate model using cross-validation"""
    classifier = PulseRateClassifier()
    
    # Initialize metrics storage
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    # Initialize confusion matrix
    cm_sum = np.zeros((2, 2))
    
    # Perform k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train the model
        classifier.train(X_train, y_train)
        
        # Make predictions
        y_pred = classifier.predict(X_test)
        
        # Calculate metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        
        # Update confusion matrix
        cm_sum += confusion_matrix(y_test, y_pred, labels=[0, 1])
    
    # Print average metrics
    print(f"\nCross-Validation Results (k={n_splits}):")
    print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_sum / n_splits, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return classifier

def main():
    # Load real sensor data
    df = load_real_data('src/data/ffffinal.csv')
    
    # Preprocess data
    processed_df = preprocess_real_data(df)
    
    # Extract features
    X, y = extract_features(processed_df)
    
    print(f"\nExtracted {len(X)} feature vectors with {X.shape[1]} features each")
    
    # Evaluate model with cross-validation
    classifier = evaluate_model_with_cross_validation(X, y)
    
    print("\nEvaluation complete. Confusion matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    main()