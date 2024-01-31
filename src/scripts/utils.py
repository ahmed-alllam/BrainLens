import requests, json

import random

import numpy as np

import torch
from torch.utils.data import DataLoader

from huggingface_hub import HfFolder
import webdataset as wds

train_dataset_urls = "pipe:curl -s -L https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/main/webdataset_avg_split/train/train_subj0{subj}_{i}.tar -H 'Authorization:Bearer {hf_token}'"
test_dataset_urls = "pipe:curl -s -L https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/main/webdataset_avg_split/test/test_subj0{subj}_{i}.tar -H 'Authorization:Bearer {hf_token}'"
meta_dataset_url = "https://huggingface.co/datasets/pscotti/naturalscenesdataset/raw/main/webdataset_avg_split/metadata_subj0{subj}.json"

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dataloader(batch_size, num_workers, split='train', num_splits=1, subjects=[0]):
    hf_token = HfFolder().get_token()

    if split == 'train':
        dataset_urls = train_dataset_urls
    else:
        dataset_urls = test_dataset_urls

    dataset_urls = [dataset_urls.format(subj=subj, i=i, hf_token=hf_token) for i in range(num_splits) for subj in subjects]

    dataset = wds.WebDataset(dataset_urls, resampled=False)\
                .decode("torch")\
                .rename(images="jpg;png", voxels="nsdgeneral.npy", trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
                .to_tuple("voxels", "images", "coco")\
                .batched(batch_size, partial=True)

    dataloader = DataLoader(dataset,
                            batch_size=None,
                            shuffle=False,
                            num_workers=num_workers)

    num = 0

    for subj in subjects:
        meta_file = requests.get(meta_dataset_url.format(subj=subj)).text
        meta = json.loads(meta_file)
        num += meta['totals'][split]

    return dataloader, num

def soft_clip_loss(voxel_embeddings, image_embeddings, temp=0.125):
    clip_clip = (image_embeddings @ image_embeddings.T)/temp
    brain_embeddings = (voxel_embeddings @ image_embeddings.T)/temp

    loss1 = -(brain_embeddings.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_embeddings.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()

    loss = (loss1 + loss2)/2

    return loss
