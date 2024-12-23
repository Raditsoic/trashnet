import torch
import torch.nn as nn
import torch.nn.functional as F

class TrashnetTorchModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(TrashnetTorchModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  
        self.fc2 = nn.Linear(128, num_classes) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(-1, 64 * 56 * 56)  
        x = F.relu(self.fc1(x))      
        x = self.fc2(x)            
        return x 