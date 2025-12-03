
from dataset import SensorDatasetKFold
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from argparse import ArgumentParser
from tools.utils import print_log, seed_everything
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Preprocessing
N_FEATURES = 12
RANDOM_STATE = 42

def rf_train(args): 
    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
    
    data = pd.read_csv(f'/home/anjilabudathoki/PinDetection/{args.data_file_name}')
    X = pd.DataFrame(data, columns=['AccX','AccY','AccZ','GyroX','GyroY','GyroZ','RotX','RotY','RotZ','MagX','MagY','MagZ'])
    y = pd.DataFrame(data, columns=['Zone']).values.ravel() 

    print(f"\nShape of Features (X): {X.shape}")
    print(f"Shape of Target (y): {y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2, 
        random_state=RANDOM_STATE,
        stratify=y  # Important to preserve class distribution
    )

    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # RandomForest classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=200,           # Number of trees
        random_state=RANDOM_STATE,
        class_weight='balanced',    # Handle class imbalance
        n_jobs=-1                    # Use all CPU cores
    )

    print("\nTraining the Random Forest Classifier...")
    rf_classifier.fit(X_train, y_train)
    print("Training complete!")

    y_pred = rf_classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy Score: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save classification report
    with open(f"{args.result_path}/classification_report_rf.txt", "w") as f:
        f.write(f"Shape of Features (X): {X.shape} and Shape of Target (y): {y.shape} \n \n")
        f.write(f"Arguments: {args} \n \n")
        f.write(f"\nAccuracy Score: {accuracy:.4f} \n \n ")
        f.write(classification_report(y_test, y_pred, zero_division=0))
        
        

    parser = ArgumentParser(description="Train SensorNet with K-Fold Cross-Validation")
    parser.add_argument(
        "--data-file-name",
        type=str,
        required=True,
        default="combined_dataset.csv",
        help="Csv file name",
    )
    parser.add_argument(
        "--combined", type=bool, default=True, help="Combined dataset or individual"
    )
    parser.add_argument(
        "--n-folds", type=int, default=5, help="Number of folds for cross-validation."
    )
    parser.add_argument(
        "--input-size", type=int, required=True, help="Input size for the model."
    )
    parser.add_argument(
        "--num-classes", type=int, required=True, help="Number of output classes."
    )
    parser.add_argument(
        "--result-path",
        type=str,
        default="detector/results/ml_model/rf",
        help="Path to the results file.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    
    args = parser.parse_args()
    seed_everything(args.seed)
    rf_train(args)
