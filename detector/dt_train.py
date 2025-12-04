from detector.dataset import SensorDatasetKFold
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification # Used for generating sample data
from argparse import ArgumentParser


# args.mean = [-2.463068664169788, 5.745093632958801, 6.987760299625468, -0.11829463171036206, -0.1660549313358302, -0.07670911360799001, 0.11442197253433209, 0.28235705368289643, 0.5442372034956304, 20.904696629213486, -19.422094881398248, -20.908379525593013]
# args.std = [2.2680630362917045, 2.317966221726892, 2.413759374534619, 0.4515026671296862, 0.4327148885859748, 0.2675990420051168, 0.1829183904888331, 0.19106603864158114, 0.4078527366701864, 11.508254056444594, 9.6160017907936, 11.425072105818726]
    
# if args.mean is not None and args.std is not None:
#             transform=lambda x: (x - np.array(args.mean)) / np.array(args.std)
#         else:
#             transform=None
transform=None
dataset_kfold = SensorDatasetKFold('/Users/anjilabudathoki/Downloads/fall-2025-course/iot/projects/data/dataset.txt', 5, transform=transform)


print(dataset_kfold)
exit()
# We create 500 samples with 12 numeric features (X) and 2 classes (y)
# This simulates the structure of your input data
N_FEATURES = 12
N_SAMPLES = 500
RANDOM_STATE = 42

X, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    n_informative=8,  # Number of features that actually contribute to the output
    n_redundant=2,
    n_classes=2,
    random_state=RANDOM_STATE
)

# Convert to a DataFrame for better visualization (optional but good practice)
feature_names = [f'Feature_{i+1}' for i in range(N_FEATURES)]
X_df = pd.DataFrame(X, columns=feature_names)
y_series = pd.Series(y, name='Target')

print("--- Data Snapshot ---")
print(X_df.head())
print(f"\nShape of Features (X): {X.shape}")
print(f"Shape of Target (y): {y.shape}")
print("---------------------")

# --- 2. Split Data into Training and Testing Sets ---

# Splitting the data helps us evaluate how well the model generalizes to unseen data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3, # Use 30% of data for testing
    random_state=RANDOM_STATE
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# --- 3. Initialize and Train the Decision Tree Classifier ---

# Instantiate the Decision Tree model
# 'criterion'='entropy' is a common choice, but 'gini' is the default and often works well
dt_classifier = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5, # Limit tree depth to prevent overfitting (a good practice)
    random_state=RANDOM_STATE
)

# Train the model using the training data
print("\nTraining the Decision Tree...")
dt_classifier.fit(X_train, y_train)
print("Training complete!")

# --- 4. Make Predictions ---

# Use the trained model to predict the class labels for the test set
y_pred = dt_classifier.predict(X_test)

# --- 5. Evaluate the Model ---

# Calculate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {accuracy:.4f}")

# Display a detailed report of classification metrics (precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Visualize the Tree (Optional but highly recommended for understanding) ---
# Note: This requires the 'graphviz' library to be installed: `pip install graphviz`
try:
    from sklearn.tree import export_graphviz
    import graphviz

    dot_data = export_graphviz(
        dt_classifier,
        out_file=None,
        feature_names=feature_names,
        class_names=['Class 0', 'Class 1'],
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    # To view the tree, you would typically save it as a PDF or image:
    # graph.render("decision_tree_visualization", view=False, format='png')
    # print("\nDecision tree visualization file generated (requires graphviz library).")

except ImportError:
    print("\n[Optional]: Install 'graphviz' to visualize the Decision Tree structure.")
