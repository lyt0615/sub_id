import numpy as np
import torch
import torch.nn as nn

from functools import partial
from timm.models.layers import DropPath
from einops import rearrange

import torch.nn.functional as F
from torch import einsum


class FeedForward(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out=None, *, dropout=0.0, f=nn.Linear, activation=nn.GELU):
        super(FeedForward, self).__init__()
        dim_out = dim_in if dim_out is None else dim_out
        
        self.net = nn.Sequential(
            f(dim_in, hidden_dim),
            activation(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            f(hidden_dim, dim_out),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
        
    def forward(self, x):
        x = self.net(x)
        return x


class Attention1d(nn.Module):
    def __init__(self, dim_in, dim_out=None, *, heads=8, dim_head=64, dropout=0.0):
        super(Attention1d, self).__init__()
        inner_dim = heads * dim_head
        dim_out = dim_in if dim_out is None else dim_out
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim_in, inner_dim * 3, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim_out),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        )
    
    def forward(self, x, mask=None):
        b, n, _ = x.shape       # 1 107 512
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv) 
        
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  
        dots = dots + mask if mask is not None else dots
        attn = dots.softmax(dim=-1)  
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)   
        out = rearrange(out, 'b h n d -> b n (h d)')  
        out = self.to_out(out) 
        
        return out, attn


class Transformer(nn.Module):
    def __init__(self, dim_in, dim_out=None, *, heads=8, dim_head=64, dim_mlp=1024, dropout=0.0, sd=0.0, attn=Attention1d, norm=nn.LayerNorm, f=nn.Linear, activation=nn.GELU):
        super(Transformer, self).__init__()
        dim_out = dim_in if dim_out is None else dim_out
        
        self.shortcut = []
        if dim_in != dim_out:
            self.shortcut.append(norm(dim_in))
            self.shortcut.append(nn.Linear(dim_in, dim_out))
        self.shortcut = nn.Sequential(*self.shortcut)
        
        self.norm1 = norm(dim_in)
        self.attn = attn(dim_in, dim_out, heads=heads, dim_head=dim_head, dropout=dropout,)
        self.sd1 = DropPath(sd) if sd > 0.0 else nn.Identity()
        
        self.norm2 = norm(dim_out)
        self.ff = FeedForward(dim_out, dim_mlp, dim_out, dropout=dropout, f=f, activation=activation)
        self.sd2 = DropPath(sd) if sd > 0.0 else nn.Identity()
        
    def forward(self, x, mask=None):
        skip = self.shortcut(x)
        x = self.norm1(x)
        x, attn = self.attn(x, mask=mask)
        x = self.sd1(x) + skip
        
        skip = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.sd2(x) + skip
        
        return x
        


from timm.models.layers import DropPath


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, groups=1, width_per_group=64, sd=0.0, **block_kwargs):
        super().__init__()

        if groups != 1 or width_per_group != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        width = int(channels * (width_per_group / 64.)) * groups

        self.shortcut = []
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut.append(nn.Conv1d(in_channels, channels, kernel_size=1, stride=stride, padding=0, bias=False))
        self.shortcut = nn.Sequential(*self.shortcut)
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv1d(in_channels, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Conv1d(width, channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.sd = DropPath(sd) if sd > 0.0 else nn.Identity()


    def forward(self, x):
        if len(self.shortcut) > 0:
            x = self.bn(x)
            x = self.relu(x)
            skip = self.shortcut(x)
        else:
            skip = self.shortcut(x)
            x = self.bn(x)
            x = self.relu(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.sd(x) + skip

        return x
    


class LNGAPBlock(nn.Module):
    def __init__(self, in_features, n_classes, **kwargs):
        super().__init__()
        self.ln = nn.BatchNorm1d(in_features)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(7)
        self.dense = nn.Linear(7 * in_features, n_classes)

    def forward(self, x):
        x = self.ln(x)
        x = self.relu(x)
        x = self.gap(x)
        x = x.view(x.size()[0], -1)
        x = self.dense(x)
        return x
    

class MLPBlock(nn.Module):
    def __init__(self, in_features, n_classes, **kwargs):
        super(MLPBlock, self).__init__()

        self.dense1 = nn.Linear(in_features, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        self.dense2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        self.dense3 = nn.Linear(512, n_classes)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.dense3(x)

        return x


class LocalAttention(nn.Module):
    def __init__(self, dim_in, dim_out=None, *, window_size=5, k=1, heads=8, dim_head=32, dropout=0.0):
        super().__init__()
        self.attn = Attention1d(dim_in, dim_out, heads=heads, dim_head=dim_head, dropout=dropout,)
        self.window_size = window_size
        self.rel_index = self.rel_distance(window_size) + window_size - 1
        self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1) * 0.02)

    def forward(self, x, mask=None):
        b, c, h = x.shape
        p = self.window_size
        n = h // p

        mask = torch.zeros(p, p, device=x.device) if mask is None else mask
        mask = mask + self.pos_embedding[self.rel_index[:, 0], self.rel_index[:, 1]]

        x = rearrange(x, 'b c (n p) -> (b n) p c', p=p)
        x, attn = self.attn(x, mask)
        x = rearrange(x, '(b n) p c -> b c (n p)', n=n, p=p)
        return x, attn


    @staticmethod
    def rel_distance(window_size):
        i = torch.tensor(np.array([[x] for x in range(window_size)]))
        d = i[None, :] - i[:, None]     
        return d            # Relative position coding


class AttentionBlockB(nn.Module):
    expansion = 4
    
    def __init__(self, dim_in, dim_out=None, *, heads=8, dim_head=64, dropout=0.0, sd=0.0, stride=1, window_size=5, k=1, norm=nn.BatchNorm1d, activation=nn.GELU, **block_kwargs):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        attn = partial(LocalAttention, window_size=window_size, k=k)
        width = dim_in // self.expansion

        self.shortcut = []
        if stride != 1 or dim_in != dim_out * self.expansion:
            self.shortcut.appen(nn.Conv1d(dim_in, dim_out * self.expansion, stride=stride, bias=False))
        self.shortcut = nn.Sequential(*self.shortcut)
        self.norm1 = norm(dim_in)
        self.relu = activation()
        self.conv = nn.Conv1d(dim_in, width, kernel_size=1, bias=False)
        self.norm2 = norm(width)
        self.attn = attn(width, dim_out * self.expansion, heads=heads, dim_head=dim_head, dropout=dropout, )
        self.sd = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x):
        if len(self.shortcut) > 0:
            x = self.norm1(x)
            x = self.relu(x)
            skip = self.shortcut(x)
        else:
            skip = self.shortcut(x)
            x = self.norm1(x)
            x = self.relu(x)

        x = self.conv(x)
        x = self.norm2(x)
        x, attn = self.attn(x)

        x = self.sd(x) + skip

        return x


class AttentionBasicBlockB(AttentionBlockB):
    expansion = 1


class Stem(nn.Module):
    def __init__(self, dim_in, dim_out, pool=True):
        super().__init__()
        self.layer0 = []
        if pool:
            self.layer0.append(nn.Conv1d(dim_in, dim_out, kernel_size=7, stride=2, padding=3, bias=False))  # 减小两倍
            self.layer0.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))    # 减小两倍
        else:
            self.layer0.append(nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False))
        self.layer0 = nn.Sequential(*self.layer0)

    def forward(self, x):
        x = self.layer0(x)
        return x            # 1000 -> 250


class ConvMSANet(nn.Module):
    def __init__(self, block1, block2, *, num_blocks, num_blocks2, heads, fc_num_layers=0, fc_dim=1024,
                 cblock=LNGAPBlock, window_size, sd=0.0, n_classes=10, stem=Stem, name, **block_kwargs):
        super().__init__()
        self.name = name
        self.fc_num_layers = fc_num_layers
        idxs = [[j for j in range(sum(num_blocks[:i]), sum(num_blocks[:i + 1]))] for i in range(len(num_blocks))]
        sds = [[sd * j / (sum(num_blocks) - 1) for j in js] for js in idxs]

        self.layer0 = stem(1, 16)
        self.layer1 = self._make_layer(block1, block2, 16, 32, num_blocks[0], num_blocks2[0], stride=1, heads=heads[0], window_size=0, sds=sds[0], **block_kwargs)
        self.layer2 = self._make_layer(block1, block2, 32 * block2.expansion, 64, num_blocks[1], num_blocks2[1], stride=2, heads=heads[1], window_size=window_size[0], sds=sds[1], **block_kwargs)
        self.layer3 = self._make_layer(block1, block2, 64 * block2.expansion, 128, num_blocks[2], num_blocks2[2], stride=2, heads=heads[2], window_size=window_size[1], sds=sds[2], **block_kwargs)
        self.layer4 = self._make_layer(block1, block2, 128 * block2.expansion, 256, num_blocks[3], num_blocks2[3], stride=2, heads=heads[3], window_size=window_size[2], sds=sds[3], **block_kwargs)

        self.classifier = []
        if cblock is MLPBlock:
            self.classifier.append(nn.AdaptiveAvgPool1d((7)))
            self.classifier.append(cblock(7 * 256 * block2.expansion, n_classes, **block_kwargs))
        else:
            self.classifier.append(cblock(256 * block2.expansion, n_classes, **block_kwargs))
        self.classifier = nn.Sequential(*self.classifier)
        
        # fc layers 
        if fc_num_layers:
            feature = self._get_feature_size()
            fc_init_dim = feature.shape[-1] * feature.shape[-2]
            self.fc = self._create_mlp_block(fc_init_dim, fc_dim=fc_dim, fc_num_layers=fc_num_layers)
            self.fc_head = nn.Linear(fc_dim, n_classes)
            
    def _get_feature_size(self):
        self.device = next(self.parameters()).device
        with torch.no_grad():
            tmp = torch.randn(1, 1, 1024).to(self.device)
            feature = self.layer4(self.layer3(self.layer2(self.layer1(self.layer0(F.interpolate(tmp, 1400))))))
        return feature

    def _create_mlp_block(self, fc_init_dim, fc_dim, fc_num_layers):
        layers = [nn.Flatten(), nn.Linear(fc_init_dim, fc_dim), nn.ReLU()]
        
        for _ in range(1, fc_num_layers):
                layers.append(nn.Linear(fc_dim, fc_dim))
                layers.append(nn.ReLU())
                
        return nn.Sequential(*layers)
    
    @staticmethod
    def _make_layer(block1, block2, in_channels, out_channels, num_block1, num_block2, stride, heads, window_size, sds, **block_kwargs):
        alt_seq = [False] * (num_block1 - num_block2 * 2) + [False, True] * num_block2
        stride_seq = [stride] + [1] * (num_block1 - 1)

        seq, channels = [], in_channels
        for alt, stride, sd in zip(alt_seq, stride_seq, sds):
            block = block1 if not alt else block2       # False: Conv, True: multi-head self-attention
            seq.append(block(channels, out_channels, stride=stride, sd=sd, heads=heads, window_size=window_size, **block_kwargs))
            channels = out_channels * block.expansion

        return nn.Sequential(*seq)

    def forward(self, x):
        x = F.interpolate(x, 1400)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.fc_num_layers:
            x = self.fc_head(self.fc(x))
        else:
            x = self.classifier(x)
        return x


def convmsa_reflection(n_classes=40, stem=True, name='ConvMSANet_Reflection', **block_kwargs):

    return ConvMSANet(BasicBlock, AttentionBasicBlockB, stem=partial(Stem, pool=stem),
                    num_blocks=(2, 2, 2, 2), num_blocks2=(0, 1, 1, 1), heads=(3, 3, 6, 12), window_size=(25, 8, 4), sd=0.1,
                    n_classes=n_classes, name=name, **block_kwargs)


# def convmsa_transmission(n_classes=42, stem=True, name='ConvMSANet_Transmission', **block_kwargs):

#     return ConvMSANet(BasicBlock, AttentionBasicBlockB, stem=partial(Stem, pool=stem),
#                     num_blocks=(2, 2, 2, 2), num_blocks2=(0, 1, 1, 1), heads=(3, 2, 4, 8), window_size=(15, 10, 5), sd=0.1,
#                     n_classes=n_classes, name=name, **block_kwargs)    

if __name__ == '__main__':
    params = {'conv_ksize':3, 
       'conv_padding':1, 
       'conv_init_dim':32, 
       'conv_final_dim':256, 
       'conv_num_layers':4, 
       'mp_ksize':2, 
       'mp_stride':2, 
       'fc_dim':1024, 
       'fc_num_layers':4, 
       'mixer_num_layers':4,
       'n_classes':957,
       'use_mixer':True
       }
    net = convmsa_reflection(**params)
    inp = torch.randn((16, 1, 1024))
    from time import time
    from thop import profile
    t1 = time()
    out = net(inp)
    t2=time()
    print(net, out.shape)
    flops, params = profile(net, inputs=(inp, ))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
