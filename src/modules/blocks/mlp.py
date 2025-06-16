import torch.nn as nn
import torch.nn.init as init

class MLP(nn.Module):
    '''
        Multi-layer perceptron
    '''
    def __init__(self, input_dim: int = 1, output_dim: int = 1, hidden_dim=None):
        '''
        :param input_dim: int
        :param output_dim: int
        :param hidden_dim: List of int
        '''
        super(MLP, self).__init__()
        if hidden_dim is None:
            hidden_dim = [32, 32, 32]
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_dim) - 2):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim[-1], output_dim))
        self.model = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = self.model(x)
        return y

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                # Initialize weights using Xavier (Glorot) initialization
                init.kaiming_normal_(layer.weight)
                # Initialize biases to zeros
                if layer.bias is not None:
                    init.zeros_(layer.bias)