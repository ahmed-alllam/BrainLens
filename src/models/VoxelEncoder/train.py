import torch

from torch import nn
from torch import optim
from torch.nn import functional as F

import tqdm

from diffusers.models import AutoencoderKL

from models import VoxelEncoder

from scripts import utils


# hyperparameters

seed = 2003

input_dim=15724
hidden_dim=4069
num_blocks=4

batch_size = 32
num_workers = 4
num_epochs = 120

lr_scheduler = 'cycle'
initial_lr = 1e-3
max_lr = 5e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_model(num_steps_per_epoch):
    model = VoxelEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks
    )

    return model

def train():
    utils.seed_everything(seed)

    voxel_image_dataloader, num_train_samples = utils.create_train_dataloader(
        batch_size=batch_size,
        num_workers=num_workers
    )

    num_steps_per_epoch = num_train_samples // batch_size

    voxel_encoder = create_model()

    diffusion_autoencoder = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256
    )

    optimizer = optim.AdamW(voxel_encoder.parameters(), lr=initial_lr)

    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=num_steps_per_epoch,
        div_factor=1e3,
        final_div_factor=1e3,
        pct_start=2/num_epochs,
    )

    for epoch in tqdm.tqdm(range(num_epochs)):
        for i, (voxel, image, _) in enumerate(voxel_image_dataloader):
            optimizer.zero_grad()

            voxel = voxel.to(device)
            image = image.to(device)

            voxel_pred = voxel_encoder(voxel)
            image_pred = diffusion_autoencoder.encode(image)
            
            loss = F.mse_loss(voxel_pred, image_pred)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            print(f'Epoch {epoch} | Batch {i} | Loss {loss.item()}')
    
    torch.save(voxel_encoder.state_dict(), '../models/autoencoder_subj01/voxel_encoder.pt')


def main():
    train()

if __name__ == '__main__':
    main()