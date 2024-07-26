"""
source codes can be found at 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(self, token_dim=768, num_filters=256, kernel_sizes=[3, 4, 5, 6], n_classes=30):
        super(TextCNN, self).__init__()

        self.mu_embedding = nn.Embedding(4000, token_dim, padding_idx=0)
        self.sigma_embedding = nn.Linear(1, token_dim)

        self.intermediate_layer = nn.Sequential(
            nn.Linear(token_dim*2, token_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(token_dim, token_dim),
            nn.Dropout(0.1)
            )
        
        self.backbone = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, token_dim)) for k in kernel_sizes
            ])

        self.head = nn.Linear(num_filters * len(kernel_sizes), n_classes)

    def forward(self, x):
        mu = self.mu_embedding(x['mus'].round().to(torch.long))
        sigma = self.sigma_embedding(x['sigmas'].unsqueeze(-1))
        
        # # weight = self.norm(self.weight_embedding(x['weights'].unsqueeze(-1)))
        # amp = x['amps'].unsqueeze(-1)
        # x = x * amp

        x = torch.cat((mu, sigma), dim=-1)
        x = self.intermediate_layer(x).unsqueeze(1)
        x = [F.relu(block(x)).squeeze(3) for block in self.backbone] 
        x = [F.avg_pool1d(item, item.size(2)).squeeze(2) for item in x]  # [batch_size, num_filters]
        x = torch.cat(x, dim=1)
        x = self.head(x)
        return x


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
