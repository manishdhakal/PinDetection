import torch.nn as nn


class SensorNet(nn.Module):
    def __init__(self, input_size: int, num_classes: int,  residual_blocks: int = 3):
        super(SensorNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()

        self.final = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(128, 64) for _ in range(residual_blocks)]
        )

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        
        for block in self.residual_blocks:
            out = block(out)

        out = self.final(out)
        out = self.softmax(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, in_features),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.layer(x)
        out += residual
        out = self.relu(out)
        return out
