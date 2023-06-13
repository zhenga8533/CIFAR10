import torch.nn as nn
import torch


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64*4*4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))  # => N, 32, 30, 30
        x = self.pool(x)  # => N, 32, 15, 15
        x = self.relu(self.conv2(x))  # => N, 64, 13, 13
        x = self.pool(x)  # => N, 64, 6, 6
        x = self.relu(self.conv3(x))  # => N, 64, 4, 4
        x = torch.flatten(x, 1)  # => N, 1024
        x = self.relu(self.fc1(x))  # => N, 64
        x = self.fc2(x)  # => N, 10
        return x
