
import torch.nn as nn
import torch
from functools import partial
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F

class Conv(nn.Module):
    default_act = nn.LeakyReLU()
    def __init__(self, c1, c2, k=1, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm1d(c2)
        self.act = self.default_act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ConvAttention(nn.Module):
    def __init__(self, in_channels, channels, reduction_factor=4):
        super(ConvAttention, self).__init__()
        inter_channels = max(in_channels//reduction_factor, 16)
        self.channels = channels
        self.fc1 = Conv(in_channels,inter_channels,1)
        self.fc2 = Conv(inter_channels,channels,1)
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        y = x
        y = F.adaptive_avg_pool1d(y, 1)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.softmax(y)
        out = y.expand_as(x) * x
        return out

class inception(nn.Module):
    def __init__(self,c1,c2):
        super(inception, self).__init__()
        self.conv1 = Conv(c1,c1,5,1,2)
        self.conv2 = Conv(c1,c1,7,1,3)
        self.conv3 = Conv(c1,c1,9,1,4)
        self.conv_2 = Conv(c1 * 3,c2,5,2,2)
        self.se = ConvAttention(c1 * 3,c1 * 3)
        self.droupt = nn.Dropout(0.3)

    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x_2 = torch.cat((x1,x2,x3),1)
        x_2 = self.se(x_2)
        x = self.conv_2(x_2)
        return x

class Bottleneck(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = inception(c1,c2)
        self.cv = nn.Sequential(
            nn.Conv1d(c1, c2, 1, 2, 0),
            nn.BatchNorm1d(c2),
            nn.ReLU()
        )
    def forward(self, x):
        y = self.cv1(x) + self.cv(x)
        return y


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, 768),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(768, dim),
        nn.Dropout(dropout)
    )

def MLPMixer1D(*, sequence_length, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
    assert (sequence_length % patch_size) == 0, 'sequence length must be divisible by patch size'
    num_patches = sequence_length // patch_size
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
    dim=patch_size*channels
    return nn.Sequential(
        Rearrange('b c (l p) -> b l (p c)', p = patch_size),
        nn.Linear(patch_size*channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )

class resunit(nn.Module):
    def __init__(self,data_channel,classes,a,layer,depth_mixer=8):
        super(resunit, self).__init__()
        self.layer = layer
        self.inplane = 8 * a       #卷积核数目
        self.conv0 = Conv(data_channel,8 * a,5,2)
        self.res_block = self.make_layer(Bottleneck,8 * a,layer)
        self.fc = nn.Linear(64 * a, classes)
        self.mlpmixer = MLPMixer1D(
            sequence_length=8,
            channels=160,
            patch_size=1,
            dim=512,
            depth=depth_mixer,
            num_classes=classes,
            expansion_factor=4,
            dropout=0.1
    )
    def forward(self,x):
        x = self.conv0(x)
        x = self.res_block(x)
        x = self.mlpmixer(x)

        return x

    def make_layer(self,block,plane,block_num):
        '''
        :param block: block模板
        :param plane: 每个模块中间运算的维度
        :param block_num: 重复次数
        '''
        block_list = []
        for i in range(block_num):
            block_list.append(block(self.inplane,plane))
        return nn.Sequential(*block_list)

if __name__ == '__main__':
    model = resunit(1,17,20,6)  #(通道数，多标签标签个数，卷积宽度倍数，残差块数）
    input = torch.randn(64,1,1024)
    print(model)
    model(input)