# Split dataset into n folds for cross-validation
import os
import pandas as pd
import argparse 

def split_dataset(csv_file:str, n_folds:int, output_dir:str) -> None:
    """
    Splits the dataset into n folds for cross-validation.

    Args:
        csv_file (string): Path to the csv file with annotations and sensor data.
        n_folds (int): Number of folds to split the dataset into.
        output_dir (string): Directory to save the fold CSV files.
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = pd.read_csv(csv_file)
    data = data.sample(frac=1).reset_index(drop=True)  # Shuffle the data

    fold_size = len(data) // n_folds
    for fold in range(n_folds):
        start_idx = fold * fold_size
        if fold == n_folds - 1:  # Last fold takes the remainder
            end_idx = len(data)
        else:
            end_idx = (fold + 1) * fold_size
        
        fold_data = data.iloc[start_idx:end_idx]
        fold_file = os.path.join(output_dir, f'{fold}.csv')
        fold_data.to_csv(fold_file, index=False)
        print(f'Saved fold {fold + 1} to {fold_file}')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset into n folds for cross-validation.')
    parser.add_argument('--csv-file', type=str, help='Path to the csv file with annotations and sensor data.')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of folds to split the dataset into.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling the data.')
    
    args = parser.parse_args()
    
    data_dir = os.path.dirname(args.csv_file)
    args.output_dir = os.path.join(data_dir, f'{args.n_folds}_folds')
    
    split_dataset(args.csv_file, args.n_folds, args.output_dir)