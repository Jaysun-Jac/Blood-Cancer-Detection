import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNmodell(nn.Module):
    def __init__(self):
        super(CNNmodell, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) #1 input channel in greyscale
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 37 * 37, 128) #Adjust based on output size after pooling
        self.fc2 = nn.Linear(128,2) # 2 output classes
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # FLatten tensors
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x