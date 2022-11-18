import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset import HLEDataset, JonathanDataset
from torch.utils.data import DataLoader
from model import SPLSS, LSQLocalization
import LRscheduler
import datetime
import yaml
from pathlib import Path
import os
import matplotlib.pyplot as plt
import torchmetrics


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import wandb

def main():
    config = load_config("config.yml")
    wandb.init(project="SSSLSquared", config=config)

    train_transform = A.Compose(
        [
            A.Resize(height=config['image_height'], width=config['image_width']),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        #keypoint_params=A.KeypointParams(format='xy')
    )
    val_transforms = A.Compose(
        [
            A.Resize(height=config['image_height'], width=config['image_width']),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        #keypoint_params=A.KeypointParams(format='xy')
    )


    run_name = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    os.mkdir("checkpoints/" + run_name)

    model = SPLSS(in_channels=1, out_channels=config['num_classes']).to(DEVICE)
    localizer = LSQLocalization(local_maxima_window=5)

    CELoss = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = LRscheduler.PolynomialLR(optimizer, config['num_epochs'])

    train_ds = JonathanDataset(base_path=config['dataset_path'], transform=train_transform, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], num_workers=2, pin_memory=True, shuffle=True)
    val_ds = JonathanDataset(base_path=config['dataset_path'], transform=val_transforms, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], num_workers=2, pin_memory=True, shuffle=True)

    wandb.watch(model)
    for epoch in range(config['num_epochs']):
        evaluate(val_loader, model, CELoss, epoch)
        visualize(val_loader, model, epoch)
        train(train_loader, CELoss, model, optimizer, epoch)

        checkpoint = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()} | model.get_statedict()
        torch.save(checkpoint, "checkpoints/" + run_name + "/model.pth.tar")

        with open("checkpoints/" + run_name + "/config.yml", 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

def load_config(path):
    return yaml.safe_load(Path(path).read_text())
    

def train(train_loader, loss_func, model, optim, epoch, log_interval=20):
    model.train()
    running_average = 0.0
    loop = tqdm(train_loader, desc="TRAINING")
    for images, gt_seg in loop:
        optim.zero_grad()

        images = images.to(device=DEVICE)
        gt_seg = gt_seg.to(device=DEVICE)

        # forward
        pred_seg = model(images)
        loss = loss_func(pred_seg.float(), gt_seg.long())
        
        loss.backward()
        optim.step()

        running_average += loss.item()
        loop.set_postfix(loss=loss.item())

    wandb.log({"Loss": running_average / len(train_loader)}, step=epoch)


def visualize(val_loader, model, epoch):
    model.eval()
    for images, gt_seg in val_loader:
        images = images.to(device=DEVICE)
        gt_seg = gt_seg.to(device=DEVICE)

        pred_seg = model(images).softmax(dim=1).argmax(dim=1)

        for i in range(images.shape[0]):
            wandb.log(
            {"Mask Prediction {0}".format(i) : wandb.Image(images[i].detach().cpu().numpy()*255, masks={
                "predictions" : {
                    "mask_data" : pred_seg[i].detach().cpu().numpy(),
                    "class_labels" : {0: "Background", 1: "Glottis", 2: "Vocalfold", 3: "Laserpoints"}
                },
                "ground_truth" : {
                    "mask_data" : gt_seg[i].detach().cpu().numpy(),
                    "class_labels" : {0: "Background", 1: "Glottis", 2: "Vocalfold", 3: "Laserpoints"}
                }
            })}, step=epoch)
        return


def evaluate(val_loader, model, loss_func, epoch):
    Accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=4)
    DICE = torchmetrics.Dice(num_classes=4)
    running_average = 0.0

    model.eval()
    loop = tqdm(val_loader, desc="EVAL")
    for images, gt_seg in loop:
        images = images.to(device=DEVICE)
        gt_seg = gt_seg.long()

        pred_seg = model(images)

        acc = Accuracy(pred_seg.softmax(dim=1).detach().cpu(), gt_seg)
        dice = DICE(pred_seg.softmax(dim=1).argmax(dim=1).detach().cpu(), gt_seg)
        loss = loss_func(pred_seg.detach().cpu(), gt_seg).item()
        running_average += loss

        loop.set_postfix({"DICE": dice, "ACC": acc, "Loss": loss})

    total_acc = Accuracy.compute()
    total_dice = DICE.compute()

    wandb.log({"Eval Loss": running_average / len(val_loader)}, step=epoch)
    wandb.log({"Eval Accuracy": total_acc}, step=epoch)
    wandb.log({"Eval DICE": total_dice}, step=epoch)

if __name__ == "__main__":
    main()