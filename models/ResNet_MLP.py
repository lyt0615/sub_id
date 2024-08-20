"""
soruce codes can be found at https://github.com/csho33/bacteria-ID
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5,
                               stride=stride, padding=2, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5,
                               stride=1, padding=2, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, hidden_sizes, num_blocks, input_dim=256,
                 in_channels=64, n_classes=30):
        super(ResNet, self).__init__()
        assert len(num_blocks) == len(hidden_sizes)
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.conv1 = nn.Conv1d(1, self.in_channels, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)

        # Flexible number of residual encoding layers
        layers = []
        strides = [1] + [2] * (len(hidden_sizes) - 1)
        for idx, hidden_size in enumerate(hidden_sizes):
            layers.append(self._make_layer(hidden_size, num_blocks[idx],
                                           stride=strides[idx]))
        self.encoder = nn.Sequential(*layers)

        self.z_dim = self._get_encoding_size()
        self.linear = nn.Sequential(nn.Linear(self.z_dim, 4800),
                nn.ReLU(),
                nn.Linear(4800, 3200),
                nn.ReLU(),
                nn.Linear(3200, 1600),
                nn.ReLU(),
                nn.Linear(1600, n_classes),)

    def encode(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.encoder(x)
        z = x.view(x.size(0), -1)
        return z

    def forward(self, x):
        z = self.encode(x)
        return self.linear(z)

    def _make_layer(self, out_channels, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(ResidualBlock(self.in_channels, out_channels,
                                        stride=stride))
            self.in_channels = out_channels
        return nn.Sequential(*blocks)

    def _get_encoding_size(self):
        """
        Returns the dimension of the encoded input.
        """
        temp = Variable(torch.rand(1, 1, self.input_dim))
        z = self.encode(temp)
        z_dim = z.data.size(1)
        return z_dim


# CNN parameters
layers = 6
hidden_size = 100
block_size = 2
hidden_sizes = [hidden_size] * layers
num_blocks = [block_size] * layers
input_dim = 1024
in_channels = 64
n_classes = 30


def resnet(n_classes):
    return ResNet(hidden_sizes, num_blocks, input_dim=input_dim,
                  in_channels=in_channels, n_classes=n_classes)



if __name__ == "__main__":
    import time
    net = resnet(n_classes=957)
    inp = torch.randn(16, 1, 1024)
    print(net)
    start = time.time()
    print(net(inp).shape)
    end = time.time()
    print((end-start)/16*1000)
    from thop import profile
    from time import time
    flops, params = profile(net, inputs=(inp, ))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
    # from thop import profile
    # flops,params = profile(net, inputs=(inp,))
    # print(flops/1e6, params/1e6)
    # # print(net)
