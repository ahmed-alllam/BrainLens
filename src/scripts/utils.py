import json
import random

import numpy as np

import torch
from torch.utils.data import DataLoader

import webdataset as wds

test_urls = "data/test/test_subj01_{i}.tar"
meta_url = "data/meta.json"

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_test_dataloader(batch_size, num_workers):
    test_urls_list = list(test_urls.format(i=i) for i in range(1))

    dataset = wds.WebDataset(test_urls_list, resampled=False)\
                .decode("torch")\
                .rename(images="jpg;png", voxels="nsdgeneral.npy", trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
                .to_tuple("voxels", "images", "coco")\
                .batched(batch_size, partial=True)

    dataloader = DataLoader(dataset, 
                            batch_size=None,
                            shuffle=False, 
                            num_workers=num_workers)
    
    num = json.load(open(meta_url))['totals']['test']

    return dataloader, num