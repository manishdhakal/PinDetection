# A utility fucntion to log training progress, save to txt file, with timestamp
from datetime import datetime
import os

import random
import numpy as np
import torch

def print_log(message: str, log_file: str = 'training_log.txt', print_output:bool = True) -> None:
    """
    Logs the training progress message to a specified log file.

    Args:
        message (string): The message to log.
        log_file (string): The file to which the log should be written.
    """
    if print_output:
        print(message)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'[{timestamp}] {message}\n')
        

def seed_everything(seed: int = 42) -> None:
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)