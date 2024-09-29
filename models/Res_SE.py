"""
soruce codes can be found at https://github.com/csho33/bacteria-ID
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
        self.channel_linear = nn.Conv1d(token_dims, token_dims, kernel_size=1)
        self.token_mixers = nn.ModuleList() 
        self.channel_mixers = nn.ModuleList() 

        for _ in range(num_layers):
            token_mixer = nn.Sequential(
                # nn.LayerNorm(token_dims),
                nn.Linear(token_dims, hidden_dims), 
                nn.GELU(), 
                nn.Dropout(dropout), 
                nn.Linear(hidden_dims, token_dims), 
                nn.Dropout(dropout)
                )
            
            channel_mixer = nn.Sequential(
                # nn.LayerNorm(num_tokens),
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
        x = self.channel_linear(x)
        for i in range(self.num_layers):
            x = x.permute(0, 2, 1)  # b, channels (token_dims), length (num_tokens) → b, num_token, token_dims
            id1 = x
            x = self.token_norm(x)
            x = self.token_mixers[i](x)
            x += id1
            
            x = self.token_norm(x)
            x = x.permute(0, 2, 1) # b, num_token, token_dims → b, token_dim, num_tokens
            id2 = x
            x = self.channel_mixers[i](x)
            x += id2 # b, token_dim, num_tokens == b, channels, length
        
        x = self.channel_norm(x)
        x = x.mean(-1) # b, channels, length → b, channels
        return x

class Conv(nn.Module):
    default_act = nn.LeakyReLU()
    def __init__(self, c1, c2, k=1, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm1d(c2)
        self.act = self.default_act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
class SE_block(nn.Module):
    def __init__(self, in_channels, channels, reduction_factor=4):
        super(SE_block, self).__init__()
        inter_channels = max(in_channels//reduction_factor, 16)
        self.channels = channels
        self.fc1 = Conv(in_channels,inter_channels,1)
        self.fc2 = Conv(inter_channels,channels,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x
        y = F.adaptive_avg_pool1d(y, 1)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        out = y.expand_as(x) * x
        return out
        
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=0):
        super(ResidualBlock, self).__init__()

        # Layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5,
                               stride=stride, padding=2, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5,
                               stride=1, padding=2, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.se = SE_block(out_channels, out_channels) if use_se else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, hidden_sizes, num_blocks, input_dim, in_channels, n_classes=957, fc_num_layers=0, fc_dim=1024, use_mixer=0, mixer_num_layers=0, use_se=0, **kwargs):
        super(ResNet, self).__init__()
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.use_mixer = use_mixer

        self.conv1 = nn.Conv1d(1, self.in_channels, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)

        # Flexible number of residual encoding layers
        layers = []
        strides = [1] + [2] * (len(hidden_sizes) - 1)
        for idx, hidden_size in enumerate(hidden_sizes):
            layers.append(self._make_layer(hidden_size, num_blocks[idx],
                                           stride=strides[idx], use_se=use_se))
        self.encoder = nn.Sequential(*layers)

        self.z_dim = self._get_encoding_size()

        # fc layers 
        self.fc_num_layers = fc_num_layers
        self.fc_dim = fc_dim
        num_tokens = self.z_dim[2]
        token_dims = self.z_dim[1]
        if self.use_mixer:
            self.mlpmixer = MLPMixer1D(num_tokens=num_tokens,
                                       token_dims=token_dims,
                                       num_layers=mixer_num_layers,
                                       hidden_dims=768,
                                       dropout=0.0)

            self.head = nn.Linear(token_dims, n_classes)
        elif fc_num_layers:
            self.fc = self._create_mlp_block(fc_init_dim=num_tokens*token_dims, fc_dim=fc_dim, fc_num_layers=fc_num_layers)
            self.fc_head = nn.Linear(fc_dim, n_classes)

        else:
            self.linear = nn.Linear(num_tokens*token_dims, self.n_classes)

        self._initialize_weights()
        
    def _create_mlp_block(self, fc_init_dim, fc_dim, fc_num_layers):
        layers = [nn.Flatten(), nn.Linear(fc_init_dim, fc_dim), nn.ReLU()]
        
        for _ in range(1, fc_num_layers):
                layers.append(nn.Linear(fc_dim, fc_dim))
                layers.append(nn.ReLU())
                
        return nn.Sequential(*layers)

    def encode(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.encoder(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        if self.use_mixer:
            z = self.head(self.mlpmixer(z))
        elif self.fc_num_layers:
            z = z.view(z.size(0), -1)
            z = self.fc_head(self.fc(z))
        else: 
            z = z.view(z.size(0), -1)
            z = self.linear(z)
        return z

    def _make_layer(self, out_channels, num_blocks, stride=1, use_se=0):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(ResidualBlock(self.in_channels, out_channels,
                                        stride=stride, use_se=use_se))
            self.in_channels = out_channels
        return nn.Sequential(*blocks)

    def _get_encoding_size(self):
        """
        Returns the dimension of the encoded input.
        """
        temp = Variable(torch.rand(2, 1, self.input_dim))
        z = self.encode(temp)
        return z.shape

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# CNN parameters
# layers = 6
# hidden_size = 100
# block_size = 2
# hidden_sizes = [hidden_size] * layers
# num_blocks = [block_size] * layers
# input_dim = 1024
# in_channels = 64
# n_classes = 30


def resnet(depth=6, hidden_size=100, block_size=2,  input_dim=1024,
                 in_channels=64, **kwargs):
    hidden_sizes = [hidden_size] * depth
    num_blocks = [block_size] * depth
    input_dim = input_dim
    in_channels = in_channels
    assert len(num_blocks) == len(hidden_sizes)
    return ResNet(hidden_sizes, num_blocks, input_dim=input_dim,
                  in_channels=in_channels, **kwargs)



if __name__ == "__main__":
    import time
    params = {'conv_ksize':3, 
              'conv_padding':1, 
              'conv_init_dim':32, 
              'conv_final_dim':256, 
              'conv_num_layers':4, 
              'mp_ksize':2, 
              'mp_stride':2, 
              'fc_dim':1024, 
              'fc_num_layers':4, 
              'mixer_num_layers':2,
              'n_classes':957,
              'use_mixer':0,
              'use_se': 1,
              'depth': 8,
              }
    net = resnet(**params)
    inp = torch.randn(16, 1, 1024)
    print(net)
    start = time.time()
    end = time.time()
    print((end-start)/16*1000)
    from thop import profile
    from time import time
    flops, params = profile(net, inputs=(inp, ))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
    # from thop import profile
    # flops,params = profile(net, inputs=(inp,))
    # print(flops/1e6, params/1e6)
    # # print(net)
