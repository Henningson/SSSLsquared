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