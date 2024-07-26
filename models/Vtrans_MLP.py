#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :vanilla_transformer.py
@Description :
@InitTime    :2024/05/10 11:07:59
@Author      :XinyuLu
@EMail       :xinyulu@stu.xmu.edu.cn
'''

import sys
sys.path.append('/data/YantiLiu/projects/substructure-ID/datasets')

from models.modules import clones, LayerNorm, EncoderLayer, MultiHeadedAttention, PositionwiseFeedForward, LearnablePositionalEncoding, LearnableClassEmbedding
from models.base import register_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class SpectralEncoding(nn.Module):
    def __init__(self, dim, patch_size, norm_layer):
        super().__init__()
        self.encoding = nn.Conv1d(
            1, dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = norm_layer(dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.encoding(x).transpose(1, 2)  # B, C, L -> B, L, C
        return self.norm(x)


class FPGrowingModule(nn.Module):
    """FPGrowingModule.

    Accept an input hidden dim and progressively grow by powers of 2 s.t.

    We eventually get to the final output size...

    """

    def __init__(
        self,
        hidden_input_dim: int = 256,
        final_target_dim: int = 4096,
        num_splits=4,
        reduce_factor=2,
    ):
        super().__init__()

        self.hidden_input_dim = hidden_input_dim
        self.final_target_dim = final_target_dim
        self.num_splits = num_splits
        self.reduce_factor = reduce_factor

        final_output_size = self.final_target_dim

        # Creates an array where we end with final_size and have num_splits + 1
        # different entries in it (e.g., num_splits = 1 with final dim 4096 has
        # [2048, 4096])
        layer_dims = [
            int(np.ceil(final_output_size / (reduce_factor**num_split)))
            for num_split in range(num_splits + 1)
        ][::-1]

        # Start by predicting into the very first layer dim (e.g., 256  -> 256)
        self.output_dims = layer_dims

        # Define initial predict module
        self.initial_predict = nn.Sequential(
            nn.Linear(
                hidden_input_dim,
                layer_dims[0],
            ),
            nn.Sigmoid(),
        )
        predict_bricks = []
        gate_bricks = []
        for layer_dim_ind, layer_dim in enumerate(layer_dims[:-1]):
            out_dim = layer_dims[layer_dim_ind + 1]

            # Need to update nn.Linear layer to be fixed if the right param is
            # called
            lin_predict = nn.Linear(layer_dim, out_dim)
            predict_brick = nn.Sequential(lin_predict, nn.Sigmoid())

            gate_bricks.append(
                nn.Sequential(
                    nn.Linear(hidden_input_dim, out_dim), nn.Sigmoid())
            )
            predict_bricks.append(predict_brick)

        self.predict_bricks = nn.ModuleList(predict_bricks)
        self.gate_bricks = nn.ModuleList(gate_bricks)

    def forward(self, hidden):
        """forward.

        Return dict mapping output dim to the

        """
        cur_pred = self.initial_predict(hidden)
        output_preds = [cur_pred]
        for _out_dim, predict_brick, gate_brick in zip(
            self.output_dims[1:], self.predict_bricks, self.gate_bricks
        ):
            gate_outs = gate_brick(hidden)
            pred_out = predict_brick(cur_pred)
            cur_pred = gate_outs * pred_out
            output_preds.append(cur_pred)
        return output_preds


@register_model
class VanillaTransformerEncoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, nlayer=6, d_model=512, d_ff=2048, nhead=8, dropout=0.1, vocab_size=1000, distillation=False):
        super().__init__()

        self_attn = MultiHeadedAttention(nhead, d_model, dropout)
        feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        layer = EncoderLayer(d_model, self_attn, feed_forward, dropout)

        self.distillation = distillation

        self.spectral_encoding = SpectralEncoding(d_model, 8, LayerNorm)
        self.positional_encoding = LearnablePositionalEncoding(
            d_model, dropout)
        self.class_encoding = LearnableClassEmbedding(
            d_model, dropout, dist=self.distillation)

        self.layers = clones(layer, nlayer)
        self.norm = LayerNorm(d_model)
        # self.proj = nn.Linear(d_model, vocab_size)
        self.proj = nn.Sequential(
                nn.Linear(d_model, 4800),
                nn.Linear(4800, 3200),
                nn.Linear(3200, 1600),
                nn.Linear(1600, vocab_size),
        )

    def forward(self, x, mask=None):
        x = self.spectral_encoding(x)
        x = self.positional_encoding(x)
        x = self.class_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)

        cls_token = x[:, 0]
        cls_out = self.proj(cls_token)

        if self.distillation:
            dist_token = x[:, -1]
            dist_out = self.proj(dist_token)
            return cls_out, dist_out
        else:
            return cls_out


if __name__ == "__main__":

    x = torch.rand(16, 1, 1024)
    model = VanillaTransformerEncoder(distillation=False)
    y = model(x)
    if type(y) == tuple:
        cls_y, dist_y = y
        print(cls_y.shape, dist_y.shape)
    else:
        print(y.shape)

    # net = FPGrowingModule()
    # print(net)
    # print(net(x).shape)
