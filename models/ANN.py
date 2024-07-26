# -*- coding: utf-8 -*-
# File  : ANN094.py
# Author: xinyuLu
# Date  : 2021/7/4

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channel, channel_attention=ChannelAttention, spatial_attention=SpatialAttention, ):
        super(CBAM, self).__init__()
        self.ca = channel_attention(in_channel)
        self.sa = spatial_attention()

    def forward(self, x):
        ca = self.ca(x)
        x = x * ca
        sa = self.sa(x)
        x = x * sa

        return x


class basic_block(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, return_attention=False):
        super(basic_block, self).__init__()
        self.return_attention = return_attention
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1,
                      bias=False),
            nn.BatchNorm1d(out_channel),
            nn.MaxPool1d(2, 2),
            nn.ReLU(inplace=False))
        self.attention = CBAM(out_channel)

    def forward(self, x):
        x = self.conv(x)

        x = self.attention(x)
        return x



class ANN(nn.Module):
    def __init__(self, n_classes=6):
        super(ANN, self).__init__()
        
        self.conv1 = basic_block(1, 16, 3)
        self.conv2 = basic_block(16, 32, 3)
        self.conv3 = basic_block(32, 64, 5)
        self.conv4 = basic_block(64, 128, 7)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(start_dim=1),
            nn.Linear(128*16, n_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        if isinstance(x, tuple): x = x[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.head(x)
        return x
