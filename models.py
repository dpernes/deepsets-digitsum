from torch import nn
from deepsetlayers import InvLinear


class MNIST_Adder(nn.Module):
    def __init__(self):
        super(MNIST_Adder, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(28*28, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 30),
            nn.ReLU(inplace=True)
        )
        self.adder = InvLinear(30, 30, reduction='sum', bias=True)
        self.output_layer = nn.Sequential(nn.ReLU(),
                                          nn.Linear(30, 1))

    def forward(self, X, mask=None):
        N, S, C, D, _ = X.shape
        h = self.feature_extractor(X.reshape(N, S, C*D*D))
        h = self.adder(h, mask=mask)
        y = self.output_layer(h)
        return y


class MNIST_AdderCNN(nn.Module):
    def __init__(self):
        super(MNIST_AdderCNN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.mlp = nn.Sequential(
            nn.Linear(2*2*64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 8),
            nn.ReLU(inplace=True)
        )
        self.adder = InvLinear(8, 1, reduction='sum', bias=True)

    def forward(self, X, mask=None):
        N, S, C, D, _ = X.shape
        h = self.feature_extractor(X.reshape(N*S, C, D, D))
        h = self.mlp(h.reshape(N, S, -1))
        y = self.adder(h, mask=mask)
        return y
