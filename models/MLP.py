"""
We convert original codes from tensorflow to pytorch
source code can be found at https://github.com/chopralab/candiy_spectrum
"""            

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_classes=17) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1024, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.45),

            nn.Linear(200, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15)
        )
    
        self.head = nn.Linear(150, n_classes)
    
    def forward(self, x):
        x = x.squeeze(1)
        x = self.encoder(x)
        x = self.head(x)
        return x

if __name__ == "__main__":
    import time
    net = MLP(30)
    inp = torch.randn((1000, 1, 1024))

    start = time.time()
    print(net(inp).shape)
    end = time.time()
    print(end-start)

