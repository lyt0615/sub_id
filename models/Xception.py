import torch 
import torch.nn as nn 
import torch.nn.init as init

class SeparableConv1dACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False):
        """Separable Convolutional Network
        If the input shape is : [2, 32, 128], and we want to get output size of [2, 64, 128] with kernel 3.
        In the normal convolutional operation, the number of parameters is:
            32 * 64 * 3
        In the separable convolution, the number of parameter is:
            1 * 1 * 3 * 32 + 1 * 32 * 64 = 3 * 32 * (1 + 64/3) round to 3 * 32 * 21, which has 3 times less number of
            parameters compared to the original operation
        """
        super(SeparableConv1dACT, self).__init__()
        padding = int((kernel_size - 1) // 2)
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, bias=bias)
        self.conv1.apply(normal_init)
        self.pointwise.apply(normal_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class XceptionStemBlock(nn.Module):
    def __init__(self, kernel, depth=1, max_dim=64):
        super(XceptionStemBlock, self).__init__()
        if max_dim == 64 or max_dim == 128:
            input_dim = [1, 32]
            output_dim = [32, 64]
        elif max_dim == 32:
            input_dim = [1, 16]
            output_dim = [16, 32]

        act = nn.LeakyReLU(0.3)
        self.depth = depth
        self.stem_1 = nn.Sequential()
        input_channel = input_dim[0]
        output_channel = output_dim[0]
        pad = int((kernel - 1) // 2)
        for i in range(2):
            self.stem_1.add_module("stem1_conv_%d" % (i + 1), nn.Conv1d(input_channel,
                                                                        output_channel,
                                                                        kernel_size=kernel,
                                                                        stride=1,
                                                                        padding=pad))
            self.stem_1.add_module("stem1_bn_%d" % (i + 1), nn.BatchNorm1d(output_channel))
            self.stem_1.add_module("stem1_act_%d" % (i + 1), act)
            input_channel = output_channel

        output_channel = output_dim[1]
        if depth == 2:
            self.stem_2 = nn.Sequential()
            for i in range(2):
                self.stem_2.add_module("stem2_conv_%d" % (i + 1), nn.Conv1d(input_channel,
                                                                            output_channel,
                                                                            kernel_size=kernel,
                                                                            stride=1,
                                                                            padding=pad))
                self.stem_2.add_module("stem2_bn_%d" % (i + 1), nn.BatchNorm1d(output_channel))
                self.stem_2.add_module("stem2_act_%d" % (i + 1), act)
                input_channel = output_channel
            self.stem_2.apply(normal_init)

        self.stem_1.apply(normal_init)

    def forward(self, x):
        x = self.stem_1(x)
        x = nn.MaxPool1d(2)(x)
        if self.depth == 2:
            x = self.stem_2(x)
        return x

    def forward_test(self, x):
        x = self.stem_1(x)
        x_pool = nn.MaxPool1d(2)(x)
        if self.depth == 2:
            x_further = self.stem_2(x_pool)
        else:
            x_further = x_pool
        return x, x_pool, x_further


class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, repeats, kernel_size,
                 stride=1, act="relu", start_with_act=True, grow_first=True,
                 separable_act=False):
        super(XceptionBlock, self).__init__()
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.skipbn = nn.BatchNorm1d(out_channels)
        else:
            self.skip = None

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "leakyrelu":
            self.act = nn.LeakyReLU(0.3, inplace=True)
        else:
            print("------The required activation function doesn't exist--------")
        self.separable_act = separable_act
        rep = []
        filters = in_channels
        if grow_first:
            rep.append(self.act)
            rep.append(SeparableConv1dACT(in_channels, out_channels, kernel_size, bias=False))
            rep.append(nn.BatchNorm1d(out_channels))
            filters = out_channels

        for i in range(repeats)[1:]:
            rep.append(self.act)
            rep.append(SeparableConv1dACT(filters, out_channels, kernel_size, bias=False))
            rep.append(nn.BatchNorm1d(out_channels))
            filters = out_channels

        if not grow_first:
            rep.append(self.act)
            rep.append(SeparableConv1dACT(filters, out_channels, kernel_size, bias=False))
            rep.append(nn.BatchNorm1d(out_channels))

        if not start_with_act:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        return x + skip


class Xception(nn.Module):
    def __init__(self, wavenumber, stem_kernel, num_xception_block=2, act="leakyrelu", depth=128,
                 stem_max_dim=64, within_dropout=False, separable_act=False, n_classes=30):
        super(Xception, self).__init__()
        self.depth = depth
        self.num_xception_block = num_xception_block
        self.stem = XceptionStemBlock(stem_kernel, 2, stem_max_dim)
        self.block1 = XceptionBlock(stem_max_dim, depth, repeats=2, kernel_size=stem_kernel,
                                    stride=1, act=act, start_with_act=False, grow_first=True,
                                    separable_act=separable_act)
        self.block2 = XceptionBlock(depth, depth, repeats=2, kernel_size=stem_kernel,
                                    stride=1, act=act, start_with_act=True, grow_first=True,
                                    separable_act=separable_act)
        if num_xception_block == 3:
            self.block3 = XceptionBlock(depth, depth, repeats=2, kernel_size=stem_kernel,
                                        stride=1, act=act, start_with_act=True, grow_first=True,
                                        separable_act=separable_act)
        if num_xception_block == 2:
            self.feature_dimension = wavenumber // 2
        elif num_xception_block == 3:
            self.feature_dimension = wavenumber // 4
        self.within_dropout = within_dropout
        self.head = nn.Sequential(
            # nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128*512, n_classes)
        )
    def forward(self, x):
        x = self.stem(x)
        if self.within_dropout:
            x = nn.Dropout(p=0.5, inplace=True)(x)
        x = self.block1(x)
        if self.num_xception_block == 3:
            x = nn.MaxPool1d(2)(x)
        if self.within_dropout:
            x = nn.Dropout(p=0.5, inplace=True)(x)
        x = self.block2(x)
        if self.num_xception_block == 3:
            x = self.block3(x)
        x = self.head(x)
        return x



def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
        init.normal_(m.weight, 0, 0.05)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def kaiming_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv1d)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


def xception(n_classes):
    return Xception(wavenumber=1024, stem_kernel=21, num_xception_block=2, act="leakyrelu",
                                  depth=128, stem_max_dim=64,
                                  within_dropout=False, separable_act=False, n_classes=n_classes)
if __name__ == "__main__":
    import time
    net = xception(30)

    inp = torch.randn((1, 1, 1024))
    start = time.time()
    print(net(inp).shape)
    end = time.time()
    print(end-start)

