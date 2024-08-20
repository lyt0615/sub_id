import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class MLPMixer1D(nn.Module):
    def __init__(self, 
                 num_tokens, 
                 token_dims, 
                 num_layers, 
                 hidden_dims=1024,
                 dropout=0.0,
                #  device='cpu'
                 ):
        super().__init__()

        self.num_layers = num_layers

        self.token_mixers = nn.ModuleList() 
        self.channel_mixers = nn.ModuleList() 

        for _ in range(num_layers):
            token_mixer = nn.Sequential(
                nn.Linear(token_dims, hidden_dims), 
                nn.GELU(), 
                nn.Dropout(dropout), 
                nn.Linear(hidden_dims, token_dims), 
                nn.Dropout(dropout)
                )
            
            channel_mixer = nn.Sequential(
                nn.Linear(num_tokens, hidden_dims), 
                nn.GELU(), 
                nn.Dropout(dropout), 
                nn.Linear(hidden_dims, num_tokens), 
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
            num_tokens = feature.shape[2]
            token_dims = feature.shape[1]
            self.mlpmixer = MLPMixer1D(num_tokens=num_tokens,
                                       token_dims=token_dims,
                                       num_layers=mixer_num_layers,
                                       dropout=0.0).to(self.device)

            self.head = nn.Linear(token_dims, n_classes)
            
        # fc layers 
        else:
            self.fc = self._create_mlp_block(fc_init_dim, fc_dim=fc_dim, fc_num_layers=fc_num_layers)
            self.head = nn.Linear(fc_dim, n_classes)
            
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
        layers = [nn.Flatten(), nn.Linear(fc_init_dim, fc_dim), nn.ReLU()]
        
        for _ in range(1, fc_num_layers):
                layers.append(nn.Linear(fc_dim, fc_dim))
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
              'fc_dim':1024, 
              'fc_num_layers':4, 
              'mixer_num_layers':4,
              'n_classes':957,
              'use_mixer':True,
              }
    net = CNN_exp(**params).to(device)
    # print(net)
    out = net(input)
    print(net)

    flops, params = profile(net, inputs=(input, ))

    print(f'FLOPs = {flops/1e9 :.4f} G')
    print(f'Params = {params/1e6 :.4f} M')