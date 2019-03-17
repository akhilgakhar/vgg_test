import torch.nn as nn
from collections import OrderedDict


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1)),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),

            ('c2', nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)),
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),

            ('c3', nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)),
            ('relu3', nn.ReLU()),
            ('c4', nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)),
            ('relu4', nn.ReLU()),
            ('s3', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),

            ('c5', nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)),
            ('relu5', nn.ReLU()),
            ('c6', nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)),
            ('relu6', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),

            ('c7', nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)),
            ('relu7', nn.ReLU()),
            ('c8', nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)),
            ('relu8', nn.ReLU()),
            ('s5', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),

        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('d1',nn.Dropout()),
            ('l1',nn.Linear(512, 512)),
            ('relu9',nn.ReLU(True)),
            ('d2',nn.Dropout()),
            ('l2',nn.Linear(512, 512)),
            ('relu10',nn.ReLU(True)),
            ('l3',nn.Linear(512, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output
