from torch import nn

from diffusers.models.autoencoders.vae import Decoder

class VoxelEncoder(nn.Module):
    def __init__(self, input_dim=15724, hidden_dim=4096, num_blocks=4):
        super(VoxelEncoder, self).__init__()

        def linear_block(in_dim, out_dim, dropout_prob=0.5):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=False),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            )

        self.linear1 = linear_block(input_dim, hidden_dim)
        self.mlp = nn.ModuleList([linear_block(hidden_dim, hidden_dim, dropout_prob=0.25) for _ in range(num_blocks)])
        
        self.linear2 = nn.Linear(hidden_dim, 64 * 16 * 16, bias=False)
        
        self.group_norm = nn.GroupNorm(1, 64)

        self.upsampling_decoder = Decoder(
            in_channels=64,
            out_channels=4,
            up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
            block_out_channels=[64, 128, 256],
            layers_per_block=1,
        )


    def forward(self, x):
        x = self.linear1(x)
        
        for layer in self.mlp:
            x = x + layer(x)
        
        x = self.linear2(x)

        x = x.view(-1, 64, 16, 16)
        x = self.group_norm(x)

        x = self.upsampling_decoder(x)

        return x
