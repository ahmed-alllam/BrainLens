import torch

from torch import nn
from torch import optim
from torch.nn import functional as F

import tqdm

from model import VoxelNet

import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.dirname(path.realpath(__file__)))))

from scripts import utils

# hyperparameters

seed = 2003

input_dim=15724
hidden_dim=4096
num_blocks=4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_model():
    model = VoxelNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
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
        subjects=[1]
    )

    voxel_net = create_model().to(device)

    for epoch in tqdm.tqdm(range(num_epochs)):
        for i, (voxel, image, _) in enumerate(voxel_image_dataloader):
            with torch.cuda.amp.autocast(False):
                pass

    print(f'Finished Training VoxelNet')

    torch.save(voxel_net.state_dict(), 'models/voxel_net_subj01/model.pt')

def main():
    train()

if __name__ == "__main__":
    main()
