import torch

from torch import nn
from torch import optim
from torch.nn import functional as F

from torchvision import transforms

from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPrior
from transformers import CLIPVisionModelWithProjection

import tqdm

from model import VoxelNet

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.dirname(path.realpath(__file__)))))

from scripts import utils

# hyperparameters

seed = 2003

input_dim=15724
clip_dim=768
hidden_dim=4096
num_blocks=4

subject = 1

batch_size = 32
num_workers = 4
num_epochs = 240

max_lr = 3e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_model():
    model = VoxelNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        clip_dim=clip_dim,
        num_blocks=num_blocks
    )

    return model

def train():
    utils.seed_everything(seed)

    voxel_image_dataloader, num_train_samples = utils.create_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        split='train',
        num_splits=1,
        subjects=[subject]
    )

    image_preprocessor = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((224, 224)),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    image_encoder = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14')
    image_encoder.to(device).eval()

    voxel_net = create_model().to(device)
    voxel_net.train()

    prior_net = DiffusionPriorNetwork(
        dim=clip_dim,
        depth=6,
        dim_head=64,
        heads=12,
    ).to(device)

    diffusion_prior = DiffusionPrior(
        prior_net,
        timesteps=100,
        cond_drop_prob=0.2,
        image_embed_dim=clip_dim,
        condition_on_text_encodings=False,
        image_embed_scale=None,
    ).to(device)

    diffusion_prior.train()

    optimizer = optim.AdamW(
        [
            {'params': voxel_net.parameters()},
            {'params': diffusion_prior.parameters()},
        ],
        lr=max_lr
    )

    num_steps_per_epoch = num_train_samples // batch_size

    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=num_epochs,
        steps_per_epoch=num_steps_per_epoch,
        final_div_factor=1e3,
        pct_start=2/num_epochs,
    )

    print(f'Starting Training VoxelNet for {num_epochs} epochs')

    for epoch in tqdm.tqdm(range(num_epochs)):
        for i, (voxel, image, _) in enumerate(voxel_image_dataloader):
            with torch.cuda.amp.autocast(False):
                optimizer.zero_grad()

                voxel = torch.mean(voxel, axis=1).to(device).float()
                image = image.to(device)

                image = image_preprocessor(image)

                image_embeddings = image_encoder(image).last_hidden_state
                image_embeddings = image_encoder.vision_model.post_layernorm(image_embeddings)
                image_embeddings = image_encoder.visual_projection(image_embeddings)[:, 0, :].squeeze(1)
                image_embeddings = torch.clamp(image_embeddings, -1.5, 1.5).float()
                image_embeddings = F.normalize(image_embeddings.flatten(1), dim=-1)

                voxel_embeddings = voxel_net(voxel)

                prior_loss = diffusion_prior(text_embed=voxel_embeddings, image_embed=image_embeddings)

                loss = prior_loss + 30 * utils.soft_clip_loss(voxel_embeddings, image_embeddings)

                loss.backward()
                optimizer.step()

                lr_scheduler.step()

    print(f'Finished Training VoxelNet')

    torch.save(voxel_net.state_dict(), 'models/voxel_net_subj01/model.pt')

def main():
    train()

if __name__ == "__main__":
    main()
