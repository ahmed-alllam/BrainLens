import json
import random

import numpy as np

import torch
from torch.utils.data import DataLoader

import webdataset as wds

train_urls = "../data/test_subj01_{i}.tar" # ToDo: change to train_subj01
meta_url = "../data/test_subj01.meta.json" # ToDo: change to the right meta file path


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_train_dataloader(batch_size, num_workers):
    train_urls = list(train_urls.format(i=i) for i in range(1))
    dataset = wds.WebDataset(train_urls).decode("torch").to_tuple("voxels", "images", "trial")

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            num_workers=num_workers,
                            shuffle=True)
    
    num = json.load(open(meta_url))['totals']['train']

    return dataloader, num