import torch.nn as nn

class SensorNet(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(SensorNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.softmax(out)
        return out
    
    
class SensorNetCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=(15,1), stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=(15,1))
        self.fc = nn.Linear(64 * 50, num_classes)
        self.conv2 = nn.Conv1d(in_channels, 64, kernel_size=(15,1), stride=1)
        
    def forward(self, x):
        x = self.conv1(x)
        print(x.shape, 'after conv')
        x = self.relu(x)
        print(x.shape, 'after relu')
        x = self.pool(x)
        print(x.shape, 'after pooling')
        x = x.flatten()
        print(x.shape, 'After flatten')
        x = self.fc(x)
        print(x.shape, 'After fc')
        return x
        