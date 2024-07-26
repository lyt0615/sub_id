"""
source codes can be found at https://github.com/ShawnYu1996/CNN-RNN-Ram
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):

    def __init__(self, token_dim=1024, n_classes=30):
        super(LSTM, self).__init__()

        self.mu_embedding = nn.Embedding(4000, token_dim)
        self.sigma_embedding = nn.Linear(1, token_dim)
        self.amp_embedding = nn.Linear(1, token_dim)
        self.weight_embedding = nn.Linear(1, token_dim)

        self.norm = nn.LayerNorm(token_dim)
        self.intermediate_layer = nn.Sequential(
            nn.Linear(token_dim*2, token_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(token_dim, token_dim),
            nn.Dropout(0.1),
        )
        self.backbone = nn.LSTM(
            input_size=token_dim,
            hidden_size=token_dim,
            num_layers=6,
            batch_first=True,
        )

        self.head = nn.Linear(token_dim, n_classes)

        self.encoder = nn.LSTM(input_size=token_dim, 
                                hidden_size=token_dim, 
                                num_layers=4,
                                bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = nn.Linear(4*token_dim, n_classes)
        
    def forward(self, x):
        # mu = self.mu_embedding(x['mus'].round().to(torch.long))
        # sigma = self.sigma_embedding(x['sigmas'].unsqueeze(-1))
        # # weight = self.weight_embedding(x['weights'].unsqueeze(-1))

        # amp = x['amps'].unsqueeze(-1)
        # x = torch.cat((mu, sigma), dim=-1)

        # x = self.intermediate_layer(x)
        # x = x * amp

        # outputs, _ = self.encoder(mu.permute(1, 0, 2)) # output, (h, c)
        # print(x.shape)
        outputs, _ = self.encoder(x.permute(1, 0, 2))
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        out = self.decoder(encoding)
        return out


if __name__ == "__main__":
    import time
    net = LSTM(n_classes=30)
    inp = torch.randn((1000, 1, 1024))

    start = time.time()
    print(net(inp).shape)
    end = time.time()
    print(end-start)

    from thop import profile
    flops, params = profile(net, inputs=(inp,))
    print(flops/1e6, params/1e6)
