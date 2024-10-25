
import torch.nn as nn
import torch
import torch.nn.functional as F
from MLPMixer import MLPMixer1D
    
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
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(510, 1)

    def forward(self, x):
        y = x
        y = F.adaptive_avg_pool1d(y, 1)
        # y = self.linear(y)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
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

class resunit(nn.Module):
    def __init__(self, data_channel=1, n_classes=957, a=20, layer=6, fc_num_layers=0, fc_dim=1024, use_mixer=False, mixer_num_layers=4, **kwargs):
        super(resunit, self).__init__()
        self.layer = layer
        self.inplane = 8 * a       #卷积核数目
        self.conv0 = Conv(data_channel,8 * a,5,2)
        self.res_block = self.make_layer(Bottleneck,8 * a,layer)
        self.fc = nn.Linear(64 * a, n_classes)
        self.use_mixer = use_mixer

        # fc layers 
        self.fc_num_layers = fc_num_layers
        self.fc_dim = fc_dim
        feature = self._get_feature_size()
        if fc_num_layers:
            
            fc_init_dim = feature.shape[-1] * feature.shape[-2]
            self.fc = self._create_mlp_block(fc_init_dim, fc_dim=fc_dim, fc_num_layers=fc_num_layers)
            self.fc_head = nn.Linear(fc_dim, n_classes)
        # mlp-mixer layers
        fc_init_dim = feature.shape[-1] * feature.shape[-2]
        
        if self.use_mixer:
            num_tokens = feature.shape[2]
            token_dims = feature.shape[1]
            self.mlpmixer = MLPMixer1D(num_tokens=num_tokens,
                                       token_dims=token_dims,
                                       num_layers=mixer_num_layers,
                                       expansion_factor=[768/token_dims, 768/num_tokens],
                                       dropout=0.0).to(self.device)

            self.head = nn.Linear(token_dims, n_classes)           

    def _get_feature_size(self):
        self.device = next(self.parameters()).device
        with torch.no_grad():
            tmp = torch.randn(64, 1, 1024).to(self.device)
            tmp = self.conv0(tmp)
            feature = self.res_block(tmp)
        return feature

    def _create_mlp_block(self, fc_init_dim, fc_dim, fc_num_layers):
        layers = [nn.Flatten(), nn.Linear(fc_init_dim, fc_dim), nn.ReLU()]
        
        for _ in range(1, fc_num_layers):
                layers.append(nn.Linear(fc_dim, fc_dim))
                layers.append(nn.ReLU())
                
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.conv0(x)
        x = self.res_block(x)
        if self.use_mixer:
            x = self.mlpmixer(x)
            x = self.head(x)
        elif self.fc_num_layers:
            x = self.fc_head(self.fc(x))
        else:
            x = x.flatten(1)
            x = self.fc(x)
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
    from thop import profile
    from torchinfo import summary
    from time import time
    params = {'conv_ksize':3, 
              'conv_padding':1, 
              'conv_init_dim':32, 
              'conv_final_dim':256, 
              'conv_num_layers':4, 
              'mp_ksize':2, 
              'mp_stride':2, 
              'fc_dim':1024, 
              'fc_num_layers':2, 
              'mixer_num_layers':1,
              'n_classes':957,
              'use_mixer':1,
              }
    inp = torch.randn(1,1,1024)
    model = resunit(**params).cpu()  #(通道数，多标签标签个数，卷积宽度倍数，残差块数）
    # summary(model, inp.shape)
    # start = time()
    # model(inp)
    # print(f'Inference time: {time()-start}.')
    start = time()
    model(inp)
    end = time()
    print(end-start)    
    # flops, params = profile(model, inputs=(inp, ))
    # model(input)
    # print(params/1e6)
