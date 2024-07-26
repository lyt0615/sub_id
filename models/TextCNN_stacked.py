"""
source codes can be found at 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(self, token_dim=768, n_classes=30):
        super(TextCNN, self).__init__()

        self.mu_embedding = nn.Embedding(4000, token_dim, padding_idx=0)
        self.sigma_embedding = nn.Linear(1, token_dim)
        self.amp_embedding = nn.Linear(1, token_dim)
        self.weight_embedding = nn.Linear(1, token_dim)

        self.norm = nn.LayerNorm(token_dim)
        self.intermediate_layer = nn.Sequential(
            nn.Linear(token_dim*2, token_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(token_dim, token_dim),
            nn.Dropout(0.1)
            )
        
        self.backbone = nn.Sequential(
            nn.Conv1d(token_dim, token_dim, 3, padding='same'),
            nn.BatchNorm1d(token_dim),
            nn.ReLU(),
            nn.Conv1d(token_dim, token_dim, 3, padding='same'),
            nn.BatchNorm1d(token_dim),
            nn.ReLU(),
            nn.Conv1d(token_dim, token_dim, 3, padding='same'),
            nn.BatchNorm1d(token_dim),
            nn.ReLU(),
            nn.Conv1d(token_dim, token_dim, 3, padding='same'),
            nn.BatchNorm1d(token_dim),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
            )

        self.head = nn.Linear(token_dim, n_classes)

    def forward(self, x):
        mu = self.mu_embedding(x['mus'].round().to(torch.long))
        sigma = self.norm(self.sigma_embedding(x['sigmas'].unsqueeze(-1)))
        
        # # weight = self.norm(self.weight_embedding(x['weights'].unsqueeze(-1)))
        # amp = x['amps'].unsqueeze(-1)
        # x = x * amp

        x = torch.cat((mu, sigma), dim=-1)
        x = self.intermediate_layer(x).permute(0, 2, 1)
        out = self.backbone(x)
        out = self.head(out)
        return out


if __name__ == "__main__":
    import time
    net = TextCNN(n_classes=30)
    inp = torch.randn((1000, 1, 1024))

    start = time.time()
    print(net(inp).shape)
    end = time.time()
    print(end-start)

    from thop import profile
    flops, params = profile(net, inputs=(inp,))
    print(flops/1e6, params/1e6)
