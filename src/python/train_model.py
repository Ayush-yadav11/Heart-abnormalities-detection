import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pulse_rate_classifier import PulseRateClassifier

def load_dataset(file_path):
    """Load the pulse rate dataset from CSV"""
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded with {len(df)} samples")
    print(f"Dataset columns: {df.columns.tolist()}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    return df

def preprocess_data(df):
    """Preprocess the dataset for training"""
    # Group data by label to create signal windows
    normal_data = df[df['label'] == 0]
    abnormal_data = df[df['label'] == 1]
    
    # Extract features directly from the data points
    # We'll create features based on statistical properties of the signals
    features = []
    labels = []
    
    # Process normal data
    features.append([normal_data['ir_value'].mean(), 
                    normal_data['ir_value'].std(),
                    normal_data['red_value'].mean(),
                    normal_data['red_value'].std(),
                    normal_data['ir_value'].max() - normal_data['ir_value'].min(),
                    normal_data['red_value'].max() - normal_data['red_value'].min(),
                    np.corrcoef(normal_data['ir_value'], normal_data['red_value'])[0, 1]])
    labels.append(0)  # Normal
    
    # Process abnormal data
    features.append([abnormal_data['ir_value'].mean(), 
                    abnormal_data['ir_value'].std(),
                    abnormal_data['red_value'].mean(),
                    abnormal_data['red_value'].std(),
                    abnormal_data['ir_value'].max() - abnormal_data['ir_value'].min(),
                    abnormal_data['red_value'].max() - abnormal_data['red_value'].min(),
                    np.corrcoef(abnormal_data['ir_value'], abnormal_data['red_value'])[0, 1]])
    labels.append(1)  # Abnormal
    
    return np.array(features), np.array(labels)

def extract_features_from_raw_data(df, sampling_rate=100):
    """Extract additional features from raw data for more robust training"""
    # Group data by label
    normal_data = df[df['label'] == 0]
    abnormal_data = df[df['label'] == 1]
    
    # Create more training samples by using sliding windows
    features = []
    labels = []
    
    # Create feature names for reference
    feature_names = [
        'ir_mean', 'ir_std', 'red_mean', 'red_std', 
        'ir_range', 'red_range', 'ir_red_correlation',
        'ir_red_ratio_mean', 'ir_red_ratio_std'
    ]
    
    # Add ratio between IR and RED values as an additional feature
    normal_ratio = normal_data['ir_value'] / normal_data['red_value']
    abnormal_ratio = abnormal_data['ir_value'] / abnormal_data['red_value']
    
    # Process normal data
    features.append([normal_data['ir_value'].mean(), 
                    normal_data['ir_value'].std(),
                    normal_data['red_value'].mean(),
                    normal_data['red_value'].std(),
                    normal_data['ir_value'].max() - normal_data['ir_value'].min(),
                    normal_data['red_value'].max() - normal_data['red_value'].min(),
                    np.corrcoef(normal_data['ir_value'], normal_data['red_value'])[0, 1],
                    normal_ratio.mean(),
                    normal_ratio.std()])
    labels.append(0)  # Normal
    
    # Process abnormal data
    features.append([abnormal_data['ir_value'].mean(), 
                    abnormal_data['ir_value'].std(),
                    abnormal_data['red_value'].mean(),
                    abnormal_data['red_value'].std(),
                    abnormal_data['ir_value'].max() - abnormal_data['ir_value'].min(),
                    abnormal_data['red_value'].max() - abnormal_data['red_value'].min(),
                    np.corrcoef(abnormal_data['ir_value'], abnormal_data['red_value'])[0, 1],
                    abnormal_ratio.mean(),
                    abnormal_ratio.std()])
    labels.append(1)  # Abnormal
    
    # Print feature information
    print("Feature names:")
    for i, name in enumerate(feature_names):
        print(f"  {i}: {name}")
    
    return np.array(features), np.array(labels), feature_names

def visualize_results(y_test, y_pred):
    """Visualize the classification results"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Normal', 'Abnormal']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Load dataset
    dataset_path = 'pulse_rate_dataset.csv'
    df = load_dataset(dataset_path)
    
    # Create classifier instance
    classifier = PulseRateClassifier()
    
    # Extract features directly from raw data
    print("Extracting features from raw data...")
    X, y, feature_names = extract_features_from_raw_data(df)
    print(f"Features extracted: {X.shape}")
    
    # Generate additional synthetic data to improve model robustness
    print("Generating additional synthetic data...")
    # Add small random variations to existing data points to create more samples
    X_synthetic = []
    y_synthetic = []
    
    for i in range(len(X)):
        # Create 5 variations of each sample
        for _ in range(5):
            # Add small random noise to features (up to 5% variation)
            noise = np.random.uniform(0.95, 1.05, size=X[i].shape)
            X_synthetic.append(X[i] * noise)
            y_synthetic.append(y[i])
    
    # Combine original and synthetic data
    X = np.vstack([X, np.array(X_synthetic)])
    y = np.append(y, y_synthetic)
    
    print(f"Total dataset size after augmentation: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train the model
    print("Training the model...")
    classifier.train(X_train, y_train)
    
    # Evaluate the model
    print("\nModel Evaluation:")
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Visualize results
    visualize_results(y_test, y_pred)
    
    # Save the trained model
    model_path = 'pulse_rate_model.joblib'
    joblib.dump(classifier, model_path)
    print(f"\nModel saved to {model_path}")
    
    return classifier

if __name__ == "__main__":
    main()