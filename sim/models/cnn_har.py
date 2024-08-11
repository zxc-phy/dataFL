import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

class CNNHar(nn.Module):
    def __init__(self, input_length=561, num_classes=6):
        super(CNNHar, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=1)
        self.pool = nn.MaxPool1d(2)
        self._input_length = input_length
        self._flattened_length = self.calculate_flattened_length()

        self.fc1 = nn.Linear(self._flattened_length, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def calculate_flattened_length(self):
        x = torch.zeros(1, 1, self._input_length)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        return x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self._flattened_length)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x