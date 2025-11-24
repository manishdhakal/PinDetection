import os

from typing import Optional

import torch
from torch.utils.data import Dataset, ConcatDataset
import pandas as pd

class SensorDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations and sensor data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.drop_columns = ['Timestamp']
        
        self.data = pd.read_csv(csv_file)
        self.data = self.data.drop(columns=self.drop_columns)
        self.transform = transform
        

    def _shuffle_data(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        sample = self.data.iloc[idx]
        features = sample.drop('Zone').values.astype('float32')
        label = sample['Zone']
        if self.transform:
            features = self.transform(features)
        return {'features': torch.FloatTensor(features), 'label': torch.tensor(label, dtype=torch.long)}

class SensorDatasetKFold:
    def __init__(self, data_root: str, n_folds:int, transform=None):
        """
        Args:
            data_root (string): Root directory containing the fold CSV files.
            n_folds (int): Total number of folds.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
    
        self.n_folds = n_folds
        self.data_root = data_root
        self.transform = transform
        
        self.folds_path = os.path.join(self.data_root, f'{self.n_folds}_folds')
        
        self.val_fold: Optional[Dataset] = None
        self.train_folds: Optional[ConcatDataset] = None
        
    def load_folds(self, fold_index:int):
        train_sets = []
        val_set = None
        
        for fold in range(self.n_folds):
            fold_file = os.path.join(self.folds_path, f'{fold}.csv')
            dataset = SensorDataset(fold_file, transform=self.transform)
            if fold == fold_index:
                val_set = dataset
            else:
                train_sets.append(dataset)
        
        self.val_fold = val_set
        self.train_folds = ConcatDataset(train_sets)
    
    def flush_folds(self):
        self.val_fold = None
        self.train_folds = None
        
    def get_train_dataset(self):
        if self.train_folds is None:
            raise ValueError("Folds not loaded. Call `load_folds(fold_index)` first.")
        return self.train_folds
    
    def get_val_dataset(self):
        if self.val_fold is None:
            raise ValueError("Folds not loaded. Call `load_folds(fold_index)` first.")
        return self.val_fold