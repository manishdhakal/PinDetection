import os
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from dataset import SensorDatasetKFold
from model import SensorNet
from tools.utils import print_log, seed_everything
from tqdm import tqdm

from argparse import ArgumentParser


class Trainer:
    def __init__(self, args):
        
        if args.mean is not None and args.std is not None:
            transform=lambda x: (x - np.array(args.mean)) / np.array(args.std)
        else:
            transform=None
        self.dataset_kfold = SensorDatasetKFold(args.data_root, args.n_folds, transform=transform)
        
        self.model = SensorNet(args.input_size, args.num_classes).to(args.device)
        
        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=args.lr)
        
        self.batch_size = args.batch_size
        self.log_path = args.log_path
        self.lr = args.lr
        self.lr_min = args.lr_min
        self.num_epochs = args.num_epochs
        self.device = args.device

        
    def train_fold(self, fold_index: int) -> torch.Tensor:
        self.dataset_kfold.load_folds(fold_index)
        train_dataset = self.dataset_kfold.get_train_dataset()
        val_dataset = self.dataset_kfold.get_val_dataset()
        
        print_log(f"Loaded {len(train_dataset)} training samples.", log_file=self.log_path)
        print_log(f"Loaded {len(val_dataset)} validation samples.", log_file=self.log_path)
    
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=16
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=16)

        steps_per_epoch = len(train_loader)
        total_steps = self.num_epochs * steps_per_epoch

        end_factor = self.lr_min / self.lr
        scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=end_factor,
            total_iters=total_steps,
        )

        self.model.train()
        step = 0

        epoch_tqdm = tqdm(range(self.num_epochs), unit="epoch")
        for epoch in epoch_tqdm:
            epoch_losses = []
            for batch in train_loader:
                features = batch["features"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                epoch_losses.append(loss.item())
                step += 1

            average_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            current_lr = self.optimizer.param_groups[0]["lr"]
            accuracy = self.evaluate(val_loader)
            message = f"Epoch [{epoch+1}/{self.num_epochs}], Step [{step}/{total_steps}], Loss: {average_epoch_loss:.4f}, LR: {current_lr:.6f}, Accuracy: {accuracy:.2f}%"
            print_log(
                message,
                log_file=self.log_path,
                print_output=False
            )
            epoch_tqdm.set_postfix_str(message)

        self.dataset_kfold.flush_folds()
        return accuracy

    def evaluate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print_log(
            f"Validation Accuracy: {100 * correct / total:.2f}%", log_file=self.log_path, print_output=False
        )
        return 100 * correct / total


if __name__ == "__main__":
    parser = ArgumentParser(description="Train SensorNet with K-Fold Cross-Validation")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing the fold CSV files.",
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
        "--batch-size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate for the optimizer.",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=0.0001,
        help="Minimum learning rate for linear decay.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs to train each fold.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="detector/logs",
        help="Path to the training log file.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use for training."
    )
    args = parser.parse_args()

    args.log_path = os.path.join(args.log_path, f"{args.n_folds}_folds.txt")

    # Feature normalization stats
    args.mean = [-2.463068664169788, 5.745093632958801, 6.987760299625468, -0.11829463171036206, -0.1660549313358302, -0.07670911360799001, 0.11442197253433209, 0.28235705368289643, 0.5442372034956304, 20.904696629213486, -19.422094881398248, -20.908379525593013]
    args.std = [2.2680630362917045, 2.317966221726892, 2.413759374534619, 0.4515026671296862, 0.4327148885859748, 0.2675990420051168, 0.1829183904888331, 0.19106603864158114, 0.4078527366701864, 11.508254056444594, 9.6160017907936, 11.425072105818726]
    
    seed_everything(args.seed)

    accuracy_list = []
    for fold_index in range(args.n_folds):
        print_log("*" * 100, log_file=args.log_path)
        print_log(f"Training fold {fold_index}/{args.n_folds}", log_file=args.log_path)
        trainer = Trainer(args)
        accuracy = trainer.train_fold(fold_index=fold_index)
        accuracy_list.append(accuracy)

    average_accuracy = sum(accuracy_list) / len(accuracy_list)
    print_log(
        f"X-validation Accuracies: {[round(acc, 2) for acc in accuracy_list]}",
        log_file=args.log_path,
    )
    print_log(
        f"Average X-validation Accuracy: {average_accuracy:.2f}%",
        log_file=args.log_path,
    )
