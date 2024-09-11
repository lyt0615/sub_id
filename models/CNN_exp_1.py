import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_exp(nn.Module):
    def __init__(self,
                 conv_ksize=3, conv_padding=1, conv_init_dim=32, conv_final_dim=64, conv_num_layers=4,
                 mp_ksize=2, mp_stride=2, 
                 fc_dim=512, fc_num_layers=4, 
                 mixer_num_layers=4,
                 n_classes=957,
                 use_mixer=False
                 ):
        
        super(CNN_exp, self).__init__()
        
        self.conv_init_dim = conv_init_dim
        self.conv_final_dim = conv_final_dim
        self.conv_num_layers = conv_num_layers
        self.use_mixer = use_mixer

        
        # convolutional layers
        conv_layers = []
        inp_channels, out_channels = self._get_conv_channels()

        for i in range(conv_num_layers):

            conv = nn.Conv1d(inp_channels[i], out_channels[i], kernel_size=conv_ksize, padding=conv_padding)
            batch_norm = nn.BatchNorm1d(out_channels[i])
            max_pool = nn.MaxPool1d(kernel_size=mp_ksize, stride=mp_stride) if mp_ksize else nn.Identity()
            activation = nn.ReLU() if i != conv_num_layers-1 else nn.Identity() # 最后一次卷积不做ReLU

            block = [conv, batch_norm, max_pool, activation]
            conv_layers.extend(block) 
                
        self.conv_layers = nn.Sequential(*conv_layers)

        # mlp-mixer layers
        feature = self._get_feature_size()
        fc_init_dim = feature.shape[-1] * feature.shape[-2]
        
        if self.use_mixer:
            num_tokens = feature.shape[-1]
            token_dims = feature.shape[-2]
            self.mlpmixer = MLPMixer1D(
            sequence_length=num_tokens,
            channels=token_dims,
            patch_size=1,
            dim=token_dims,
            depth=mixer_num_layers,
            num_classes=n_classes,
            expansion_factor=2,
            dropout=0.1
    )
            
        # fc layers 
        else:
            self.fc = self._create_mlp_block(fc_init_dim, fc_dim=fc_dim, fc_num_layers=fc_num_layers)
            self.head = nn.Linear(fc_dim, n_classes) if fc_num_layers else nn.Linear(fc_init_dim, n_classes)
            
    def _get_feature_size(self):
        self.device = next(self.parameters()).device
        with torch.no_grad():
            tmp = torch.randn(1, 1, 1024).to(self.device)
            feature = self.conv_layers(tmp)
        return feature
    
    def _get_conv_channels(self):
        # if self.conv_num_layers <= 4:
        #     out_channels = np.linspace(self.conv_init_dim, self.conv_final_dim, self.conv_num_layers).astype(int)
        #     out_channels = list(out_channels)
        # else:
        #     out_channels = np.linspace(self.conv_init_dim, self.conv_final_dim, self.conv_num_layers).astype(int)
        #     out_channels = list(out_channels)
        #     out_channels += [256 for _ in range(self.conv_num_layers - 4)]

        out_channels = [256 for _ in range(self.conv_num_layers)]
        # out_channels = [np.ceil(self.conv_init_dim * ((self.conv_final_dim / self.conv_init_dim) ** (1 / (self.conv_num_layers - 1))) ** i).astype(int) for i in range(self.conv_num_layers)]
        inp_channels = [1] + out_channels[:-1] 
        return inp_channels, out_channels

    def _create_mlp_block(self, fc_init_dim, fc_dim, fc_num_layers):
        if fc_num_layers>=1:
            layers = [nn.Flatten(), nn.Linear(fc_init_dim, fc_dim), nn.ReLU()]
            
            for _ in range(1, fc_num_layers):
                    layers.append(nn.Linear(fc_dim, fc_dim))
                    layers.append(nn.ReLU())
        if fc_num_layers == 0:
            layers = [nn.Flatten()]
                
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv_layers(x)

        if self.use_mixer:
            x = self.mlpmixer(x)
        else:
            x = self.fc(x)
        return x


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim*expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim*expansion_factor, dim),
        nn.Dropout(dropout)
    )
from einops.layers.torch import Rearrange, Reduce
from functools import partial

def MLPMixer1D(*, sequence_length, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
    assert (sequence_length % patch_size) == 0, 'sequence length must be divisible by patch size'
    num_patches = sequence_length // patch_size
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
    # dim=patch_size*channels

    return nn.Sequential(
        Rearrange('b c (l p) -> b l (p c)', p = patch_size),
        nn.Linear(patch_size*channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, 64, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, 4, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )

if __name__ == '__main__':
    from thop import profile
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    torch.manual_seed(1)
    input = torch.randn(1, 1, 1024).to(device)
    p= []
    for f in range(1, 8):
        params = {'conv_ksize':3, 
                'conv_padding':1, 
                'conv_init_dim':32, 
                'conv_final_dim':256, 
                'conv_num_layers':6, 
                'mp_ksize':2, 
                'mp_stride':2, 
                'fc_dim':1024, 
                'fc_num_layers':1, 
                'mixer_num_layers':f,
                'n_classes':957,
                'use_mixer':1,
                }
        net = CNN_exp(**params).to(device)
        tb_writer = SummaryWriter(log_dir = 'checkpoints/qm9s_raman/CNN_exp/net')
        tb_writer.add_graph(net, (input))
        # print(net)
        out = net(input)
        print(net)

        flops, params = profile(net, inputs=(input, ))

        print(f'FLOPs = {flops/1e9 :.4f} G')
        print(f'Params = {params/1e6 :.4f} M')
        p.append(params/1e6)
    print(p)