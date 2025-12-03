from dataset import SensorDatasetKFold
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from argparse import ArgumentParser
from tools.utils import print_log, seed_everything
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Preprocessing
N_FEATURES = 12
RANDOM_STATE = 42


def hyperparams_grid_search(args, X_train_scaled, y_train, X_test_scaled, y_test):
    param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf'], 
    'class_weight': ['balanced']
    }
    grid_search = GridSearchCV(
    estimator=SVC(random_state=RANDOM_STATE),
    param_grid=param_grid,
    scoring='f1_weighted',
    cv=3, 
    verbose=2, 
    n_jobs=-1  )
    grid_search.fit(X_train_scaled, y_train)

    print("\n--- ✅ Grid Search Complete ---")
    print(f"Best parameters found on training data: {grid_search.best_params_}")
    print(f"Best Cross-Validation F1-score: {grid_search.best_score_:.4f}")

    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test_scaled)

    print("\n--- Final Model Evaluation (Best SVM) ---")
    print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Test Set Classification Report:")
    print(classification_report(y_test, y_pred))

def svm_train(args): 
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
        stratify=y  # Makes sure that the class distribution in the train and test set matches the original dataset.
    )

    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    
    
    scaler = StandardScaler() # When we did not scale,accuracy was less than 15 for all the dataset except manish
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    hyperparams_grid_search(args,X_train_scaled, y_train, X_test_scaled, y_test=y_test )
    exit()

    svm_classifier = SVC(
        C=1.0, 
        kernel='rbf', 
        gamma='scale',
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )

    print("\nTraining the Support Vector Machine...")
    svm_classifier.fit(X_train_scaled, y_train)
    print("Training complete!")

    y_pred = svm_classifier.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy Score: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
   
    
    with open(f"{args.result_path}/classification_report.txt", "w") as f:
        f.write(f"Shape of Features (X): {X.shape} and Shape of Target (y): {y.shape} \n \n")
        f.write(f"Arguments: {args} \n \n")
        f.write(f"\nAccuracy Score: {accuracy:.4f} \n \n ")
        f.write(classification_report(y_test, y_pred))


if __name__ == "__main__":
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
        default="detector/results/ml_model/",
        help="Path to the results file.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    
    args = parser.parse_args()
    seed_everything(args.seed)
    svm_train(args)