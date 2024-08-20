import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops.layers.torch import Rearrange, Reduce
# from functools import partial


    
# class PreNormResidual(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.fn = fn
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, x):
#         return self.fn(self.norm(x)) + x

# def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
#     return nn.Sequential(
#         dense(dim, dim * expansion_factor),
#         nn.GELU(),
#         nn.Dropout(dropout),
#         dense(dim * expansion_factor, dim),
#         nn.Dropout(dropout)
#     )


# def MLPMixer1D(*, sequence_length, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
#     assert (sequence_length % patch_size) == 0, 'sequence length must be divisible by patch size'
#     num_patches = sequence_length // patch_size
#     chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

#     return nn.Sequential(
#         Rearrange('b c (l p) -> b l (p c)', p = patch_size),
#         nn.Linear(patch_size*channels, dim),
#         *[nn.Sequential(
#             PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
#             PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
#         ) for _ in range(depth)],
#         nn.LayerNorm(dim),
#         Reduce('b n c -> b c', 'mean'),
#         nn.Linear(dim, num_classes)
#     )          



# def create_mlp_block(input_dim, output_dim, num_layers):
#     layers = []
#     # current_dim = input_dim
#     # interval = (input_dim - output_dim) // num_layers
    
#     # for i in range(num_layers):
#         # if i != num_layers-1:
#         #     if input_dim - (i+1) * interval <= output_dim:
#         #         next_output_dim = current_dim
#         #     else:
#         #         next_output_dim = input_dim - (i+1) * interval
#         #     layers.append(nn.Linear(current_dim, next_output_dim))
#         #     layers.append(nn.ReLU())
#         #     current_dim = next_output_dim
#         # else: 
#         #     next_output_dim = output_dim
#         #     layers.append(nn.Linear(current_dim, next_output_dim))
#     for i in range(num_layers):
#         if num_layers == 1:
#             layers.append(nn.Linear(input_dim, output_dim))
#         elif i == 0:
#             layers.append(nn.Linear(input_dim, output_dim))
#             layers.append(nn.ReLU())
#         elif i != num_layers-1:
#             layers.append(nn.Linear(output_dim, output_dim))
#             layers.append(nn.ReLU())
#         else:
#             layers.append(nn.Linear(output_dim, output_dim))
#     return nn.Sequential(*layers)

# def getModelSize(model):
    # param_size = 0
    # param_sum = 0
    # for param in model.parameters():
    #     param_size += param.nelement() * param.element_size()
    #     param_sum += param.nelement()
    # buffer_size = 0
    # buffer_sum = 0
    # for buffer in model.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()
    #     buffer_sum += buffer.nelement()
    # all_size = (param_size + buffer_size) / 1024 / 1024
    # return all_size


# if n_conv == 1:
#     conv_param = [{'conv_cin': 1, 'conv_cout': 256, 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,}]
# else:
#     conv_param = [{'conv_cin': 1, 'conv_cout': 32, 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,}]
#     series = [int(32 * ((256 / 32) ** (1 / (n_conv - 1))) ** i) for i in range(n_conv)]
#     for i in range(1, n_conv):
#         conv_param.append({'conv_cin': series[i-1], 'conv_cout': series[i], 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,})
# fc_param = get_fc_param([args.batch_size, 1024], conv_param, n_fc)
# net = CNN_exp(conv_param, fc_param, depth_mixer, n_classes=n_classes).to(device)


# def get_fc_param(input_size, conv_param, fc_layers, dropout=None):
#     for params in conv_param:
#         input_size[1] = (input_size[1]+2*params['conv_padding']-params['conv_ksize'])//params['conv_stride']+1
#         input_size[0] = params['conv_cout']
#         if params['mp_ksize'] is not None and params['mp_stride'] is not None:
#             input_size[1] = (input_size[1]-params['mp_ksize'])//params['mp_stride']+1
#     return {'input_dim': input_size[0]*input_size[1], 'num_layers': fc_layers, 'dropout': dropout}



class MLPMixer1D(nn.Module):
    def __init__(self, 
                 num_tokens, 
                 token_dims, 
                 num_layers, 
                 expansion_factor=4, 
                 dropout=0.0,
                #  device='cpu'
                 ):
        super().__init__()

        self.num_layers = num_layers

        self.token_mixers = nn.ModuleList() 
        self.channel_mixers = nn.ModuleList() 

        for _ in range(num_layers):
            token_mixer = nn.Sequential(
                nn.Linear(token_dims, token_dims * expansion_factor), 
                nn.GELU(), 
                nn.Dropout(dropout), 
                nn.Linear(token_dims * expansion_factor, token_dims), 
                nn.Dropout(dropout)
                )
            
            channel_mixer = nn.Sequential(
                nn.Linear(num_tokens, num_tokens * expansion_factor), 
                nn.GELU(), 
                nn.Dropout(dropout), 
                nn.Linear(num_tokens * expansion_factor, num_tokens), 
                nn.Dropout(dropout)
                )
            
            self.token_mixers.append(token_mixer)
            self.channel_mixers.append(channel_mixer)

        self.token_norm = nn.LayerNorm(token_dims)
        self.channel_norm = nn.LayerNorm(num_tokens)

    def forward(self, x):

        for i in range(self.num_layers):
            x = x.permute(0, 2, 1)  # b, channels (token_dims), length (num_tokens) → b, num_token, token_dims
            id1 = x 
            x = self.token_mixers[i](x)
            x = self.token_norm(x)
            x += id1
            
            x = x.permute(0, 2, 1) # b, num_token, token_dims → b, token_dim, num_tokens
            id2 = x
            x = self.channel_mixers[i](x)
            x = self.channel_norm(x)
            x += id2 # b, token_dim, num_tokens == b, channels, length
        
        x = self.channel_norm(x)
        x = x.mean(-1) # b, channels, length → b, channels
        return x



class CNN_exp(nn.Module):
    def __init__(self,
                 conv_ksize=3, conv_padding=1, conv_init_dim=32, conv_final_dim=64, conv_num_layers=4,
                 mp_ksize=2, mp_stride=2, 
                 fc_output_dim=957, fc_num_layers=4, 
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
            num_tokens = feature.shape[2]
            token_dims = feature.shape[1]
            self.mlpmixer = MLPMixer1D(num_tokens=num_tokens,
                                       token_dims=token_dims,
                                       num_layers=mixer_num_layers,
                                       expansion_factor=1, dropout=0.0).to(self.device)

            self.head = nn.Linear(token_dims, n_classes)
            
        # fc layers 
        else:
            self.fc = self._create_mlp_block(fc_init_dim, fc_output_dim=fc_output_dim, fc_num_layers=fc_num_layers)
            self.head = nn.Linear(fc_output_dim, n_classes)
            
    def _get_feature_size(self):
        self.device = next(self.parameters()).device
        with torch.no_grad():
            tmp = torch.randn(1, 1, 1024).to(self.device)
            feature = self.conv_layers(tmp)
        return feature
    
    def _get_conv_channels(self):
        if self.conv_num_layers <= 4:
            out_channels = np.linspace(self.conv_init_dim, self.conv_final_dim, self.conv_num_layers).astype(int)
            out_channels = list(out_channels)
        else:
            out_channels = np.linspace(self.conv_init_dim, self.conv_final_dim, self.conv_num_layers).astype(int)
            out_channels = list(out_channels)
            out_channels += [256 for _ in range(self.conv_num_layers - 4)]
        # out_channels = [np.ceil(self.conv_init_dim * ((self.conv_final_dim / self.conv_init_dim) ** (1 / (self.conv_num_layers - 1))) ** i).astype(int) for i in range(self.conv_num_layers)]
        inp_channels = [1] + out_channels[:-1] 
        return inp_channels, out_channels

    def _create_mlp_block(self, fc_init_dim, fc_output_dim, fc_num_layers):
        layers = [nn.Flatten(), nn.Linear(f.c_init_dim, fc_output_dim), nn.ReLU()]
        
        for _ in range(1, fc_num_layers):
                layers.append(nn.Linear(fc_output_dim, fc_output_dim))
                layers.append(nn.ReLU())
                
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv_layers(x)

        if self.use_mixer:
            x = self.mlpmixer(x)
        else:
            x = self.fc(x)

        x = self.head(x)
        return x




if __name__ == '__main__':
    from thop import profile
    import numpy as np
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    torch.manual_seed(1)
    input = torch.randn(1, 1, 1024).to(device)

    params = {'conv_ksize':3, 
              'conv_padding':1, 
              'conv_init_dim':32, 
              'conv_final_dim':256, 
              'conv_num_layers':4, 
              'mp_ksize':2, 
              'mp_stride':2, 
              'fc_output_dim':1024, 
              'fc_num_layers':16, 
              'mixer_num_layers':4,
              'n_classes':957,
              'use_mixer':False,
              }
    net = CNN_exp(**params).to(device)
    # print(net)
    out = net(input)
    print(out.shape)

    flops, params = profile(net, inputs=(input, ))

    print(f'FLOPs = {flops/1e9 :.4f} G')
    print(f'Params = {params/1e6 :.4f} M')


    # n_conv = 5
    # n_fc = None
    # depth_mixer = None
    # size = []
    # for n_conv in range(5, 6):
    #     for depth_mixer in (8,12,16,20):
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

            # if n_conv == 1:
            #     conv_param = [{'conv_cin': 1, 'conv_cout': 256, 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,}]
            # else:
            #     conv_param = [{'conv_cin': 1, 'conv_cout': 32, 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,}]
            #     series = [int(32 * ((256 / 32) ** (1 / (n_conv - 1))) ** i) for i in range(n_conv)]
            #     for i in range(1, n_conv):
            #         conv_param.append({'conv_cin': series[i-1], 'conv_cout': series[i], 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,})
            
            # conv_param = [{'conv_cin': 1, 'conv_cout': 32, 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,},
            #               {'conv_cin': 32, 'conv_cout': 32, 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,},
            #               {'conv_cin': 32, 'conv_cout': 32, 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,},
            #               {'conv_cin': 32, 'conv_cout': 32, 'conv_ksize': 3, 'conv_stride': 1, 'conv_padding': 1, 'mp_ksize': 2, 'mp_stride': 2,}
            #               ]
    #         fc_param = get_fc_param([64, 1024], conv_param, n_fc)
    #         net = CNN_exp(conv_param, fc_param, depth_mixer, n_classes=957).to(device)
    #         flops, params = profile(net, inputs=(torch.randn(64,1,1024).cuda(), ))
            
    #         # print('FLOPs = ' + str(flops/1000**3) + 'G')
    #         # print('Params = ' + str(params/1000**2) + 'M')
    #         size.append(params/1000**2)
    # print(size)

