import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierNet(nn.Module):
    def __init__(self, in_dim, out_classes, hidden_layers=None):
        super().__init__()
        if hidden_layers:
            first_layer_size = hidden_layers[0]
            second_layer_size = hidden_layers[1]
        else:
            first_layer_size = in_dim + 3
            second_layer_size = in_dim + 2
        self.fcin = nn.Linear(in_dim, first_layer_size)
        self.fc1 = nn.Linear( first_layer_size, second_layer_size)
        self.fcout = nn.Linear(second_layer_size, out_classes)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fcin(x))
        x = F.relu(self.fc1(x))
        x = self.fcout(x)
        return x

Net = ClassifierNet