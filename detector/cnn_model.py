import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Model(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate flattened features size dynamically
        self.flattened_size = 128 * (100 // 2 // 2) # 128 channels * 25 time steps

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)
        # Conv1d expects (batch_size, num_features, sequence_length)
        x = x.permute(0, 2, 1) 
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = x.reshape(x.size(0), -1) # Flatten the output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

if __name__ == "__main__":
    # Example usage
    model = CNN_Model(input_channels=12, num_classes=10)
    model.eval()
    model.load_state_dict(torch.load("cnn_model.pth", map_location=torch.device('cpu')))
    