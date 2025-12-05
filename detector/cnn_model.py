import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Model(nn.Module):
    def __init__(self, input_channels, num_classes, dropout=0.3):
        super(CNN_Model, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=15, padding='same')
        self.bn1   = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2 = nn.Conv1d(64, 128, kernel_size=9, padding='same')
        self.bn2   = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Block 3 - NEW DEEPER LAYER
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding='same')
        self.bn3   = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Global Average Pooling (better than flatten + reduces parameters)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # -> (B, 256, 1)
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: (batch_size, 100, n_features)
        x = x.permute(0, 2, 1)  # -> (batch_size, n_features, 100)

        # Conv Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # Conv Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Conv Block 3 (new!)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Global pooling instead of flatten (more robust)
        x = self.global_pool(x)        # -> (B, 256, 1)
        x = x.view(x.size(0), -1)      # -> (B, 256)
        
        # Classifier
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # logits

        return x
    

if __name__ == "__main__":
    model = CNN_Model(input_channels=12, num_classes=10)
    model.eval()
    model.load_state_dict(torch.load("cnn_model.pth", map_location=torch.device('cpu')))
    