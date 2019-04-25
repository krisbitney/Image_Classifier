from torch import nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_sizes):
        # make sure module is instantiated with at least one hidden layer
        assert hidden_layer_sizes, 'This module requires at least one hidden layer'
        super(Classifier, self).__init__()
        
        self.hidden_layers = nn.ModuleList([
            nn.Linear(input_size, hidden_layer_sizes[0]), 
            *[nn.Linear(i, o) for i, o in zip(hidden_layer_sizes[0:-1], hidden_layer_sizes[1:])]
            ])   
        
        self.dropout = nn.Dropout(0.2)     
        
        self.output_layer = nn.Linear(hidden_layer_sizes[-1], output_size)

        self.class_to_idx = None

    def forward(self, x):
        # unpack hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        # output layer
        x = F.log_softmax(self.output_layer(x), dim=1)
        return x