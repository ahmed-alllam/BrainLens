from torch import nn

class VoxelNet(nn.Module):
    """
    Model used in the high-level pipeline.
    It transforms the voxel representation into a CLIP embedding.
    """
    def __init__(self, input_dim=15724, hidden_dim=4096, clip_dim=768, num_blocks=4):
        super(VoxelNet, self).__init__()

        def linear_block(in_dim, out_dim, dropout_prob=0.5):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout_prob)
            )

        self.linear1 = linear_block(input_dim, hidden_dim)
        self.mlp = nn.ModuleList([linear_block(hidden_dim, hidden_dim, dropout_prob=0.15) for _ in range(num_blocks)])

        self.linear2 = nn.Linear(hidden_dim, clip_dim, bias=True)

    def forward(self, x):
        x = self.linear1(x)

        for layer in self.mlp:
            x = x + layer(x)

        x = self.linear2(x)

        return x
