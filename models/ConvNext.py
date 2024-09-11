
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x) 
        x = x.permute(0, 2, 1) # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x) 
        x = self.act(x)
        x = self.pwconv2(x) 
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1) # (N, L, C) -> (N, C, L)

        x = input + self.drop_path(x)
        return x
    

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        n_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=1, n_classes=30, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,fc_dim=1024,fc_num_layers=0, **kwargs
                 ):
        super().__init__()
        self.fc_num_layers = fc_num_layers
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv1d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv1d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], n_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

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
            feature = self.forward_features(F.adaptive_avg_pool1d(tmp, 256))
        return feature

    def _create_mlp_block(self, fc_init_dim, fc_dim, fc_num_layers):
        layers = [nn.Flatten(), nn.Linear(fc_init_dim, fc_dim), nn.ReLU()]
        
        for _ in range(1, fc_num_layers):
                layers.append(nn.Linear(fc_dim, fc_dim))
                layers.append(nn.ReLU())
                
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean(-1)) # global average pooling, (N, C, L) -> (N, C)

    def forward(self, x):
        x = F.adaptive_avg_pool1d(x, 256)
        x = self.forward_features(x)
        if self.fc_num_layers:
            x = self.fc_head(self.fc(x))
        else:
            x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x

@register_model
def convnext_tiny(**kwargs):
    model = ConvNeXt(**kwargs)
    return model

if __name__ == "__main__":
    params = {
        'conv_ksize':3, 
              'conv_padding':1, 
              'conv_init_dim':32, 
              'conv_final_dim':256, 
              'conv_num_layers':4, 
              'mp_ksize':2, 
              'mp_stride':2, 
              'fc_dim':1024, 
              'fc_num_layers':3, 
              'mixer_num_layers':4,
              'n_classes':957,
              'use_mixer':True,
              }
    net = convnext_tiny(**params)
    inp = torch.randn((16, 1, 256))
    from time import time
    from thop import profile
    t1 = time()
    out = net(inp)
    t2=time()
    print((t2-t1)/16*1000)
    flops, params = profile(net, inputs=(inp, ))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
    print(net)
