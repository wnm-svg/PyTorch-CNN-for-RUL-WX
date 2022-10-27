from torch.nn import Linear, ReLU, Sequential, Conv2d, AvgPool2d, Module, BatchNorm2d
import torch
from torchsummary import summary

class CNN1(Module):
    def __init__(self, nf):
        super(CNN1, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 8, kernel_size=(4, nf), stride=1),
            BatchNorm2d(8),
            ReLU(inplace=True),
            AvgPool2d(kernel_size=(2, 1), stride=2),
            # Defining another 2D convolution layer
            Conv2d(8, 14, kernel_size=(3, 1), stride=1),
            BatchNorm2d(14),
            ReLU(inplace=True),
            AvgPool2d(kernel_size=(2, 1), stride=2),
        )

        self.linear_layers = Sequential(
            Linear(14 * 2 * 1,1)
           
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        # for onnx: x.size(0) -> int(x.size(0))
        x = x.view(int(x.size(0)), -1) 
        x = self.linear_layers(x)
        return x


if __name__ == "__main__":
    model = CNN1(15)
    model.cuda()
    summary(model, input_size=[(1,15,15)])


