import torch

from torch import nn
from torch import optim
from torch.nn import functional as F

import tqdm

from diffusers.models import AutoencoderKL

from model import VoxelAutoEncoder

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.dirname(path.realpath(__file__)))))

from scripts import utils


# hyperparameters

seed = 2003

input_dim=15724
hidden_dim=4069
num_blocks=4

batch_size = 64
num_workers = 4
num_epochs = 120

lr_scheduler = 'cycle'
initial_lr = 1e-3
max_lr = 5e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_model():
    model = VoxelAutoEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks
    )

    return model

def train():
    utils.seed_everything(seed)

    # voxel_image_dataloader, num_train_samples = utils.create_test_dataloader(  # ToDo: Change to train, once downloaded
        # batch_size=batch_size,
    #     num_workers=num_workers
    # )
    num_train_samples = 1
    num_steps_per_epoch = 1

    # num_steps_per_epoch = num_train_samples // batch_size

    voxel_auto_encoder = create_model()

    diffusion_autoencoder = AutoencoderKL.from_single_file("https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors")

    diffusion_autoencoder.requires_grad_(False)
    diffusion_autoencoder.eval()

    optimizer = optim.AdamW(voxel_auto_encoder.parameters(), lr=initial_lr)

    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        epochs=num_epochs,
        max_lr=max_lr,
        steps_per_epoch=num_steps_per_epoch,
        div_factor=1e3,
        final_div_factor=1e3,
        pct_start=2/num_epochs,
    )

    voxel_auto_encoder.to(device)
    diffusion_autoencoder.to(device)

    print(f'Starting Training VoxelEncoder')
    print("voxel_auto_encoder device: ", next(voxel_auto_encoder.parameters()).device)
    print("diffusion_autoencoder device: ", next(diffusion_autoencoder.parameters()).device)

    for epoch in tqdm.tqdm(range(num_epochs)):
        for i, (voxel, image, _) in enumerate(voxel_image_dataloader):
            with torch.cuda.amp.autocast(False):

                voxel = torch.mean(voxel, axis=1).to(device).float()
                image = image.to(device)

                voxel_pred = voxel_auto_encoder(voxel)

                image_upsampled = F.interpolate(image, (512, 512), mode='bilinear', align_corners=False, antialias=True)
                image_pred = diffusion_autoencoder.encode(2 * image_upsampled - 1).latent_dist.mode() * 0.18215
            
                loss = F.l1_loss(voxel_pred, image_pred)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

    print(f'Finished Training VoxelAutoEncoder')
    
    torch.save(voxel_auto_encoder.state_dict(), 'models/autoencoder_subj01/voxel_auto_encoder.pt')

def main():
    train()

if __name__ == '__main__':
    main()
