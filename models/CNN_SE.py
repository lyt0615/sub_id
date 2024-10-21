
import torch.nn as nn
import torch

import torch.nn.functional as F


class MLPMixer1D(nn.Module):
    def __init__(self, 
                 num_tokens, 
                 token_dims, 
                 num_layers, 
                 hidden_dims=1024,
                 dropout=0.0,
                 ):
        super().__init__()

        self.num_layers = num_layers
        self.channel_linear = nn.Conv1d(token_dims, token_dims, kernel_size=1)
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
        # self.channel_norm = nn.LayerNorm(num_tokens)

    def forward(self, x):
        x = self.channel_linear(x)
        for i in range(self.num_layers):
            id1 = x
            x = x.permute(0, 2, 1)  # b, channels (token_dims), length (num_tokens) → b, num_token, token_dims
            x = self.token_norm(x)
            x = x.permute(0, 2, 1)
            x = self.channel_mixers[i](x)
            x += id1
            
            x = x.permute(0, 2, 1) # b, num_token, token_dims → b, token_dim, num_tokens
            id2 = x
            x = self.token_norm(x)
            x = self.token_mixers[i](x)
            x += id2
            x = x.permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        x = self.token_norm(x)
        x = x.mean(-2)
        return x

    # def forward(self, x):
    #     x = self.channel_linear(x)
    #     for i in range(self.num_layers):
    #         x = x.permute(0, 2, 1)  # b, channels (token_dims), length (num_tokens) → b, num_token, token_dims
    #         id1 = x
    #         x = self.token_norm(x)
    #         x = self.token_mixers[i](x)
    #         x += id1
            
    #         x = self.token_norm(x)
    #         x = x.permute(0, 2, 1) # b, num_token, token_dims → b, token_dim, num_tokens
    #         id2 = x
    #         x = self.channel_mixers[i](x)
    #         x += id2 # b, token_dim, num_tokens == b, channels, length
        
    #     x = self.channel_norm(x)
    #     x = x.mean(-1) # b, channels, length → b, channels
    #     return x
    
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

    def forward(self, x):
        y = x
        y = F.adaptive_avg_pool1d(y, 1)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        out = y.expand_as(x) * x
        return out

class SE_block(nn.Module):
    def __init__(self,c1,c2,use_se):
        super(SE_block, self).__init__()
        self.conv_2 = Conv(c2,c2)
        self.conv = Conv(c1, c2, 5, 2, 2)
        self.se = ConvAttention(c2,c2) if use_se else nn.Identity()
        self.droupt = nn.Dropout(0.3)

    def forward(self,x):
        x = self.conv(x)
        x = self.se(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, use_res, use_se):
        super().__init__()
        self.se_block = SE_block(c1,c2, use_se)
        self.use_res = use_res
        if use_res:
            self.res_conv = nn.Sequential(
            nn.Conv1d(c1, c2, 1, 2, 0),
            nn.BatchNorm1d(c2),
            nn.ReLU()
        ) 
    def forward(self, x):
        y = self.se_block(x) + self.res_conv(x) if self.use_res else self.se_block(x)
        return y

class resunit(nn.Module):
    def __init__(self, data_channel=1, n_classes=957, a=32, depth=6, fc_num_layers=0, fc_dim=1024, use_mixer=False, use_res=False, use_se=False, mixer_num_layers=4, **kwargs):
        super(resunit, self).__init__()
        self.layer = depth
        self.inplane = data_channel       #卷积核数目
        self.res_block = self.make_layer(Bottleneck,8 * a, depth, use_res, use_se)
        self.use_mixer = use_mixer

        # fc layers 
        self.fc_num_layers = fc_num_layers
        self.fc_dim = fc_dim
        feature = self._get_feature_size()
        self.fc = nn.Linear(feature.shape[-1] * feature.shape[-2], n_classes)
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
                                       hidden_dims=fc_dim,
                                       dropout=0.0).to(self.device)

            self.head = nn.Linear(token_dims, n_classes)           

    def _get_feature_size(self):
        self.device = next(self.parameters()).device
        with torch.no_grad():
            tmp = torch.randn(64, 1, 1024).to(self.device)
            feature = self.res_block(tmp)
        return feature

    def _create_mlp_block(self, fc_init_dim, fc_dim, fc_num_layers):
        layers = [nn.Flatten(), nn.Linear(fc_init_dim, fc_dim), nn.ReLU()]
        
        for _ in range(1, fc_num_layers):
                layers.append(nn.Linear(fc_dim, fc_dim))
                layers.append(nn.ReLU())
                
        return nn.Sequential(*layers)
    
    def forward(self,x):
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

    def make_layer(self,block,plane,block_num,use_res, use_se):
        '''
        :param block: block模板
        :param plane: 每个模块中间运算的维度
        :param block_num: 重复次数
        '''
        block_list = []
        for i in range(block_num):
            if i == 0:
                block_list.append(block(self.inplane,plane,use_res, use_se))
            else:
                block_list.append(block(plane,plane,use_res, use_se))
        return nn.Sequential(*block_list)

def inf_time(model):
    iterations = 300
    device = torch.device("cuda:0")
    model.to(device)

    random_input = torch.randn(1, 1, 1024).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # Preheat GPU
    for _ in range(50):
        _ = model(random_input)

    # Measure inference time
    times = torch.zeros(iterations)     # Save the time of each iteration
    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            _ = model(random_input)
            ender.record()
            # Synchronize GPU time
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # Calculate time
            times[iter] = curr_time

if __name__ == '__main__':
    from thop import profile
    from torch.utils.tensorboard import SummaryWriter
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    p = []
    t = []
    for l in range(1,8):
        sum_p=0
        sum_t = 0
        params = {'conv_ksize':3, 
                'conv_padding':1, 
                'conv_init_dim':32, 
                'conv_final_dim':256, 
                'conv_num_layers':4, 
                'mp_ksize':2, 
                'mp_stride':2, 
                'fc_dim':1024, 
                'fc_num_layers':0, 
                'mixer_num_layers':l,
                'n_classes':957,
                'use_mixer':1,
                'depth': 6,
                'use_se': 0,
                'use_res': 0,
                }
        model = resunit(**params)  #(通道数，多标签标签个数，卷积宽度倍数，残差块数）
        input = torch.randn(1,1,1024)
        # tb_writer = SummaryWriter(log_dir = 'checkpoints/qm9s_raman/CNN_SE/net')
        # tb_writer.add_graph(model, (input))
        flops, params = profile(model, inputs=(input, ))
        p.append(params/1000**2)
        # t.append(inf_time(model))
    print(p,t)