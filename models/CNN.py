'''
4 convs & 4 fcs, no dropout
'''


import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

'''
6.26
1. padding去掉效果如何
# 2. focal/bce
3. fclayer:1~4
4. 调换relu/pool位置
5. kernel_size 5,7,9,11
'''
'''
6.28
1. PeakNet
2. 增加mlp层数，观察何时性能趋于饱和
3. 前五名+4*MLP
4. 可视化MLP权重
'''
class CNN(nn.Module):
    def __init__(self, class_num=37, init_weights=True):

        self.kernel_size = 3

        super(CNN, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=self.kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=self.kernel_size, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=self.kernel_size)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=self.kernel_size)

        # Define the fully connected layers
        # self.fc1 = nn.Linear(15*1024, 4800)
        self.fc1 = nn.Linear(15872, 4800)  # 15360, 14848, 14336, 14080
        self.fc2 = nn.Linear(4800, 3200)
        self.fc3 = nn.Linear(3200, 1600)
        self.fc4 = nn.Linear(1600, class_num)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)

        # Define activation functions and pooling layer
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout()

        # Define the output activation function (e.g., sigmoid for binary classification)
        self.sigmoid = nn.Sigmoid()

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten the tensor before passing it through fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.fc4(x)

        # x = self.sigmoid(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

import torch
import torch.nn.functional as F

from torchvision.utils import _log_api_usage_once


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: torch.Tensor = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(sigmoid_focal_loss)
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    # if all(alpha) >= 0:
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

if __name__ == "__main__":
    from thop import profile
    from time import time
    input = torch.randn((16,1,1024))
    net = CNN()
    t1 = time()
    output = net(input)
    t2=time()
    print((t2-t1)/16*1000)
    flops, params = profile(net, inputs=(input, ))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')