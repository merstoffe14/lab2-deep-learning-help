import torch
import torch.nn as nn

from options.classification_options import ClassificationOptions


class Print(nn.Module):
    """"
    This model is for debugging purposes (place it in nn.Sequential to see tensor dimensions).
    """

    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)
        return x


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        """START TODO: replace None with a Linear layer"""
        #I couldn't get this linear layer to work with 20,1 for example but it works with 1,1
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
        """END TODO"""

    def forward(self, x: torch.Tensor):
        """START TODO: forward the tensor x through the linear layer and return the outcome (replace None)"""
        x = self.linear_layer(x)
        """END TODO"""
        return x

class Classifier(nn.Module):
    def __init__(self, options: ClassificationOptions):
        super().__init__()
        """ START TODO: fill in all three layers. 
            Remember that each layer should contain 2 parts, a linear layer and a nonlinear activation function.
            Use options.hidden_sizes to store all hidden sizes, (for simplicity, you might want to 
            include the input and output as well).
        """
        #hidden_sizes ??

        #ik heb hier nu willekeurige in en out features genomen.
        #test eens wat er gebeurt als de in featerus van layer 2 niet gelijk is aan de out features van layer 1
        

        # Linear and ReLu (28*28 pixels so  784 in_features)
        self.layer1 = nn.Sequential(
            nn.Linear(in_features= 28*28, out_features= 50),
            nn.ReLU()
        )
        # Linear and ReLu
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=50, out_features= 20),
            nn.ReLU()
            
        )
        # Linear and softmax (softmax normalizes the last layer, so every value between [0 1] and sum(layer) = 1)
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=20, out_features=10),
            nn.Softmax(dim=1)
        )
        """END TODO"""

    def forward(self, x: torch.Tensor):
        """START TODO: forward tensor x through all layers."""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        """END TODO"""
        return x


class ClassifierVariableLayers(nn.Module):
    def __init__(self, options: ClassificationOptions):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(len(options.hidden_sizes) - 1):
            self.layers.add_module(
                f"lin_layer_{i + 1}",
                nn.Linear(options.hidden_sizes[i], options.hidden_sizes[i + 1])
            )
            if i < len(options.hidden_sizes) - 2:
                self.layers.add_module(
                    f"relu_layer_{i + 1}",
                    nn.ReLU()
                )
            else:
                self.layers.add_module(
                    f"softmax_layer",
                    nn.Softmax(dim=1)
                )
        print(self)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        return x
