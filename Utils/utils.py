import torch
import torchvision
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_config(path):
    return yaml.safe_load(Path(path).read_text())

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def generate_weights_from_dataset(dataloader, num_classes):
    seg_list = []
    for _, gt_seg, _ in dataloader:
        seg_list.append(gt_seg.detach().cpu().numpy())

    segs = np.concatenate(seg_list, axis=0).astype(np.uint8)
    num_classes = 4
    elements_of_classes = []
    for i in range(num_classes):
        elements_of_classes.append(np.count_nonzero(segs == i))

    weights = []
    for i in range(num_classes):
        weights.append(max(elements_of_classes) / elements_of_classes[i])

    return weights

def normalize_image_batch(images: torch.tensor) -> torch.tensor:
    channelwise_minima = images.min(2, keepdim=True)[0].min(3, keepdim=True)[0]
    channelwise_maxima = images.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
    return (images - channelwise_minima) / (channelwise_maxima - channelwise_minima)

if __name__ == "__main__":
    a = torch.randn((4, 3, 512, 512))
    norm_a = normalize_image_batch(a)