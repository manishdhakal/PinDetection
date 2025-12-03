import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# Import the K-Nearest Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler 
from collections import Counter 


RANDOM_STATE = 42
K_NEIGHBORS = 5 

data = pd.read_csv('/home/anjilabudathoki/PinDetection/detector/data/anjila_dataset.csv')
X = data[['AccX','AccY','AccZ','GyroX','GyroY','GyroZ','RotX','RotY','RotZ','MagX','MagY','MagZ']]
y = data['Zone'].values.ravel() 

print(f"\nFull Dataset Class Distribution: {Counter(y)}")
print("---------------------------------------")

# --- 2. Split Data (Using Stratified Sampling) ---

# Stratified split is used to ensure the class distribution is preserved.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3, 
    random_state=RANDOM_STATE,
    stratify=y 
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Training Set Class Distribution: {Counter(y_train)}")
print(f"Testing Set Class Distribution: {Counter(y_test)}")


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_classifier = KNeighborsClassifier(
    n_neighbors=K_NEIGHBORS,
    weights='uniform' 
)

print(f"\nTraining the K-Nearest Neighbors (K={K_NEIGHBORS})...")
knn_classifier.fit(X_train_scaled, y_train)
print("Training complete!")


# Use the trained model to predict the class labels for the scaled test set
y_pred = knn_classifier.predict(X_test_scaled)

# Calculate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {accuracy:.4f}")

# Display a detailed report of classification metrics
print("\nClassification Report (K-Nearest Neighbors):")
print(classification_report(y_test, y_pred))