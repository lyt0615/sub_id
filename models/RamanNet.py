"""
we convert sorce codes from tensorflow to pytorch 
sorce code can be found at https://github.com/nibtehaz/RamanNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RamanNet(nn.Module):
    def __init__(self, inp_len=1024, w_len=64, dw=32, n_classes=30):
        super().__init__()

        self.w_len = w_len
        self.dw = dw
        n_windows = (inp_len - w_len) // dw

        self.stems = nn.ModuleList([])
        for _ in range(n_windows):
            self.stems.append(
                nn.Sequential(
                    nn.Linear(w_len, dw),
                    nn.BatchNorm1d(dw),
                    nn.LeakyReLU(inplace=True)
                )
            )

        self.backbone = nn.Sequential(
            nn.Linear(n_windows * dw, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(512, 256)
        )

        self.head = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = x.squeeze(1)
        features = []
        for i, stem in enumerate(self.stems):
            inp = x[:, i*self.dw:i*self.dw+self.w_len]
            features.append(stem(inp))
        features = torch.cat(features, dim=1)
        features = F.dropout(features, 0.5)
        features = self.backbone(features)

        out = self.head(features)
        return out 

if __name__ == "__main__":
    import time
    net = RamanNet(n_classes=30)
    inp = torch.randn((1000, 1, 1024))
    start = time.time()
    print(net(inp).shape)
    end = time.time()
    print(end-start)

