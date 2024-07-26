import torch
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

def MLPMixer1D(*, sequence_length, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
    assert (sequence_length % patch_size) == 0, 'sequence length must be divisible by patch size'
    num_patches = sequence_length // patch_size
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (l p) -> b l (p c)', p = patch_size),
        nn.Linear(patch_size*channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )



if __name__ == '__main__':
    # Example input: batch_size=32, channels=1, length=128
    x = torch.randn(64, 256, 56)

    # Example usage
    sequence_length = x.shape[-1]  # Example sequence length
    channels = x.shape[1]
    patch_size = 8        # Example patch size
    dim = 512               # Example dimensionality
    depth = 12              # Example depth
    num_classes = 957       # Example number of classes

    mixer = MLPMixer1D(
        sequence_length=sequence_length,
        channels=channels,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        num_classes=num_classes,
        expansion_factor=4,
        dropout=0.1
    )
    output = mixer(x)
    print(output.shape)  # Should be (32, 10)
