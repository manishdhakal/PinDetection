from dataset import SensorDatasetKFold
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from argparse import ArgumentParser
from tools.utils import seed_everything
import matplotlib.pyplot as plt
import os

N_FEATURES = 12
RANDOM_STATE = 42

def barplot(args, bar_data, save_file_name, title):
    plt.figure(figsize=(6,4))
    bar_data.plot(kind="bar")
    plt.title(f"{title} Distribution")
    plt.xlabel("Zone")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f'{args.result_path}{save_file_name}')
    plt.close()   

    

def plots(data, y_train, y_test, args):
    os.makedirs(args.result_path, exist_ok=True)
    
    barplot(args, data['Zone'].value_counts(), 'full.png', 'Full data')
    barplot(args, pd.Series(y_train).value_counts(), 'train.png', 'Training set')
    barplot(args, pd.Series(y_test).value_counts(), 'test.png', 'Test set')


def dt_train(args):


    data = pd.read_csv(f'/home/anjilabudathoki/PinDetection/{args.data_file_name}')
    X = pd.DataFrame(data, columns=['AccX','AccY','AccZ','GyroX','GyroY','GyroZ','RotX','RotY','RotZ','MagX','MagY','MagZ'])
    y= pd.DataFrame(data, columns=['Zone']).values.ravel() 

    print(f"\nShape of Features (X): {X.shape}")
    print(f"Shape of Target (y): {y.shape}")
    print("---------------------" )



    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2, 
        random_state=RANDOM_STATE
    )

    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")


    plots(data, y_train, y_test, args)
    


    dt_classifier = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=args.max_depth_dt, 
        random_state=RANDOM_STATE
    )

    # Train the model using the training data
    print("\nTraining the Decision Tree...")
    dt_classifier.fit(X_train, y_train)
    print("Training complete!")

    # Use the trained model to predict the class labels for the test set
    y_pred = dt_classifier.predict(X_test)


    # Calculate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy Score: {accuracy:.4f}")

    # Display a detailed report of classification metrics (precision, recall, f1-score)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    with open(f"{args.result_path}/classification_report.txt", "w") as f:
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
    
    parser.add_argument(
        "--max-depth-dt", type=int, default=5, help="Max depth for decision tree"
    )
    args = parser.parse_args()
    seed_everything(args.seed)
    dt_train(args)