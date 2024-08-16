import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from functools import partial


pair = lambda x: x if isinstance(x, tuple) else (x, x)

def get_fc_param(input_size, conv_param, fc_layers, dropout=None):
    for params in conv_param:
        input_size[1] = (input_size[1]+2*params['conv_padding']-params['conv_ksize'])//params['conv_stride']+1
        input_size[0] = params['conv_cout']
        if params['mp_ksize'] is not None and params['mp_stride'] is not None:
            input_size[1] = (input_size[1]-params['mp_ksize'])//params['mp_stride']+1
    return {'input_dim': input_size[0]*input_size[1], 'num_layers': fc_layers, 'dropout': dropout}

class CNN_exp(nn.Module):
    def __init__(self, conv_param, fc_param, depth_mixer, n_classes=957):
        super(CNN_exp, self).__init__()

        # convolutional layers
        conv_layers = []
        for param in conv_param:
            if param['mp_ksize'] is not None and param['mp_stride'] is not None:  
                block = [nn.Conv1d(param['conv_cin'], param['conv_cout'], kernel_size=param['conv_ksize'], stride=param['conv_stride'], padding=param['conv_padding']),
                        nn.MaxPool1d(kernel_size=param['mp_ksize'], stride=param['mp_stride']),
                        nn.BatchNorm1d(param['conv_cout']),
                        nn.ReLU()
                        ]
            else: 
                block = [nn.Conv1d(param['conv_cin'], param['conv_cout'], kernel_size=param['conv_ksize'], stride=param['conv_stride'], padding=param['padding']),
                        nn.BatchNorm1d(param['conv_cout']),
                        nn.ReLU()
                        ]
            conv_layers.extend(block)
        self.conv_layers = nn.Sequential(*conv_layers)

        # fc layers
        if fc_param['num_layers'] is not None:
            self.fc= create_mlp_block(fc_param['input_dim'], output_dim=n_classes, num_layers=fc_param['num_layers'])
        else:
            self.mlpmixer = MLPMixer1D(
            sequence_length=int(fc_param['input_dim']/param['conv_cout']),
            channels=param['conv_cout'],
            patch_size=2,
            dim=957,
            depth=depth_mixer,
            num_classes=n_classes,
            expansion_factor=1,
            dropout=0.1)
        self.depth_mixer = depth_mixer
    
    def forward(self, x):
        x = self.conv_layers(x)
        if self.depth_mixer is not None:
            x = self.mlpmixer(x)
        else:
            x = x.view(x.size(0), -1)
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
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )
def MLPMixer1D(*, sequence_length, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
    assert (sequence_length % patch_size) == 0, 'sequence length must be divisible by patch size'
    num_patches = sequence_length // patch_size
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

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
    
def create_mlp_block(input_dim, output_dim, num_layers):
    layers = []
    # current_dim = input_dim
    # interval = (input_dim - output_dim) // num_layers
    
    # for i in range(num_layers):
        # if i != num_layers-1:
        #     if input_dim - (i+1) * interval <= output_dim:
        #         next_output_dim = current_dim
        #     else:
        #         next_output_dim = input_dim - (i+1) * interval
        #     layers.append(nn.Linear(current_dim, next_output_dim))
        #     layers.append(nn.ReLU())
        #     current_dim = next_output_dim
        # else: 
        #     next_output_dim = output_dim
        #     layers.append(nn.Linear(current_dim, next_output_dim))
    for i in range(num_layers):
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        elif i == 0:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
        elif i != num_layers-1:
            layers.append(nn.Linear(output_dim, output_dim))
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Linear(output_dim, output_dim))
    return nn.Sequential(*layers)

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    return all_size

if __name__ == '__main__':
    from thop import profile
    import numpy as np
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_conv = 5
    n_fc = None
    depth_mixer = None
    size = []
    for n_conv in range(4,5):
        for n_fc in range(4, 22, 4):
    #         if n_conv == 1:
    #             conv_param = [{'conv_cin': 1, 'conv_cout': 256, 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,}]
    #         else:
    #             conv_param = [{'conv_cin': 1, 'conv_cout': 32, 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,}]
    #             series = [int(32 * ((256 / 32) ** (1 / (n_conv - 1))) ** i) for i in range(n_conv)]
    #             for i in range(1, n_conv):
    #                 conv_param.append({'conv_cin': series[i-1], 'conv_cout': series[i], 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,})
            
    #         # conv_param = [{'conv_cin': 1, 'conv_cout': 32, 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,},
    #         #               {'conv_cin': 32, 'conv_cout': 32, 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,},
    #         #               {'conv_cin': 32, 'conv_cout': 32, 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,},
    #         #               {'conv_cin': 32, 'conv_cout': 32, 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,}
    #         #               ]
    #         fc_param = get_fc_param([1, 1024], conv_param, n_fc)
    #         net = CNN_exp(conv_param, fc_param, n_classes=957).cuda()
    #         flops, params = profile(net, inputs=(torch.randn(1,1,1024).cuda(), ))
            
    #         print('FLOPs = ' + str(flops/1000**3) + 'G')
    #         print('Params = ' + str(params/1000**2) + 'M')
    #         size[f'{n_conv}+{n_fc}'] = params/1000**2

    # print(size)

            if n_conv == 1:
                conv_param = [{'conv_cin': 1, 'conv_cout': 256, 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,}]
            else:
                conv_param = [{'conv_cin': 1, 'conv_cout': 32, 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,}]
                series = [int(32 * ((256 / 32) ** (1 / (n_conv - 1))) ** i) for i in range(n_conv)]
                for i in range(1, n_conv):
                    conv_param.append({'conv_cin': series[i-1], 'conv_cout': series[i], 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,})
            
            # conv_param = [{'conv_cin': 1, 'conv_cout': 32, 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,},
            #               {'conv_cin': 32, 'conv_cout': 32, 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,},
            #               {'conv_cin': 32, 'conv_cout': 32, 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,},
            #               {'conv_cin': 32, 'conv_cout': 32, 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,}
            #               ]
            fc_param = get_fc_param([64, 1024], conv_param, n_fc)
            net = CNN_exp(conv_param, fc_param, depth_mixer, n_classes=957).to(device)
            flops, params = profile(net, inputs=(torch.randn(64,1,1024).cuda(), ))
            
            print('FLOPs = ' + str(flops/1000**3) + 'G')
            print('Params = ' + str(params/1000**2) + 'M')
            size.append(params/1000**2/64)
    print(size)


    # [0.139037953125, 0.254009953125, 0.368981953125, 0.483953953125, 0.598925953125] 957 * 1
    # [0.145712953125, 0.277594953125, 0.409476953125, 0.541358953125, 0.673240953125] 512 * 4
    # [0.290022234375, 0.347322609375, 0.404622984375, 0.461923359375, 0.519223734375] mlp: 957