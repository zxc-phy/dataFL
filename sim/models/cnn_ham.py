import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNHAM(nn.Module):
    def __init__(self):
        super(CNNHAM, self).__init__()
        # Adjusted layer configurations
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Smaller kernel, less stride
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Standard pooling
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Increase channels, maintain dimensions
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Standard pooling
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Increase channels
        # self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # Same dimensions
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Adjusted for new flattened size
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 7)
        # self.dropout2 = nn.Dropout(0.5)
        # self.fc3 = nn.Linear(128, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc3(x)
        return F.log_softmax(x, dim=1)