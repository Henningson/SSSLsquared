import torch
import albumentations as A
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime
import yaml
from printer import Printer
import os
import argparse
import pygit2
import utils
from albumentations.pytorch import ToTensorV2
import ConfigArgsParser
import torch.nn.functional as F
import dataset
import sys
import random
import Args

sys.path.append("models/")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# From https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook.
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def f_beta_loss(pred, gt):
    smooth = 1.
    beta = 2.
    beta_sq = torch.square(beta)

    p_flat = pred.view(-1)
    g_flat = gt.view(-1)

    intersection = (p_flat * g_flat).sum()
    g_backslash_p = ((1 - p_flat) * g_flat).sum()
    p_backslash_g = (p_flat * (1 - g_flat)).sum()

    f_beta = (((1 + beta_sq) * intersection + smooth) / (((1 + beta_sq)*intersection) + (beta_sq * g_backslash_p)
                                                         + p_backslash_g + smooth)).mean()
    return 1 - f_beta


def main():
    parser = Args.GlobalArgumentParser(
                    prog = 'Keypoint Regularized Training for Semantic Segmentation',
                    description = 'Train a Segmentation Network that is optimized for simultaneously outputting keypoints',
                    epilog = 'Arguments can be used to overwrite values in a config file.')
    
    args = parser.parse_args()
    if args.checkpoint_name:
        CHECKPOINT_PATH = args.checkpoint_name
    else:
        CHECKPOINT_PATH = "sharan"

    CONFIG_PATH = args.config
    config = ConfigArgsParser.ConfigArgsParser(utils.load_config(CONFIG_PATH), args)
    train_transform =  A.load("train_transform.yaml", data_format='yaml')

    val_transforms =  A.load("val_transform.yaml", data_format='yaml')

    neuralNet = __import__(config["model"])
    model = neuralNet.Model(config=config).to(DEVICE)

    try:
        os.mkdir(CHECKPOINT_PATH)
    except:
        print("Path " + CHECKPOINT_PATH + " does already exist. Overwriting.")

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=config['learning_rate'])
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    train_ds = dataset.SharanHLE(config=config, is_train=True, transform=train_transform)
    val_ds = dataset.SharanHLE(config=config, is_train=False, transform=val_transforms)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], num_workers=config['num_workers'], pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], num_workers=config['num_workers'], pin_memory=True, shuffle=False)

    for epoch in range(config['last_epoch'], config['num_epochs']):
        # Train the network
        train(train_loader, model, optimizer, epoch)

        checkpoint = {"optimizer": optimizer.state_dict(), "optimizer": optimizer.state_dict()} | model.state_dict()
        torch.save(checkpoint, CHECKPOINT_PATH + "/model.pth.tar")

        config["last_epoch"] = epoch
        with open(CHECKPOINT_PATH + "/config.yml", 'w') as outfile:
            yaml.dump(dict(config), outfile, default_flow_style=False)

    Printer.OKG("Training Done!")

def train(train_loader, model, optimizer, epoch):
    Printer.Header("EPOCH: {0}".format(epoch))
    model.train()
    running_average = 0.0
    loop = tqdm(train_loader, desc="TRAINING")
    mse = nn.MSELoss()
    dice = DiceLoss()


    for images, gt_seg, gt_heatmap, _ in loop:
        optimizer.zero_grad()

        images = images.to(device=DEVICE)
        gt_seg = gt_seg.to(device=DEVICE)
        gt_heatmap = gt_heatmap.to(device=DEVICE)

        # forward
        binary, logits = model(images)

        stage2_loss = mse(binary[:, 0], gt_heatmap) + dice(binary[:, 0], gt_heatmap)
        stage1_loss = mse(logits[:, 0], gt_heatmap) + dice(logits[:, 0], gt_heatmap)
        loss = 0.5 * stage1_loss + 0.5 * stage2_loss
        loss.backward()
        optimizer.step()

        running_average += loss.item()
        loop.set_postfix(loss=loss.item())

if __name__ == "__main__":
    main()