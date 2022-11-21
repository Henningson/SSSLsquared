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
import argparse
import pygit2
import Visualizer
import numpy as np
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import wandb

def main():
    parser = argparse.ArgumentParser(
                    prog = 'Train a Deep Neural Network for Semantic Segmentation',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')
    parser.add_argument("-l", "--logwandb", action="store_true")
    
    args = parser.parse_args()
    LOG_WANDB = args.logwandb

    config = load_config("config.yml")
    checkpoint_name = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    os.mkdir("checkpoints/" + checkpoint_name)

    train_transform = A.Compose(
        [
            A.Resize(height=config['image_height'], width=config['image_width']),
            A.Affine(translate_percent = 0.1, p=0.25), #This leads to errors in combination with others
            A.OpticalDistortion(border_mode = cv2.BORDER_CONSTANT, shift_limit=0.7, distort_limit = 0.7, p = 0.5), #also play with shift_limit = 0.05 or distort_limit = 0.05
            A.Rotate(limit=60, border_mode = cv2.BORDER_CONSTANT, p=0.5), #Border Mode Constant for not duplicating the plica vocalis
            A.HorizontalFlip(p=0.5), #0.5
            A.VerticalFlip(p=0.25), #0.1
            A.RandomBrightnessContrast(contrast_limit = [-0.10, 0.6],p=0.5),
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

    if LOG_WANDB:
        repo = pygit2.Repository('.')
        num_uncommitted_files = repo.diff().stats.files_changed

        if num_uncommitted_files > 0:
            print("\033[93m" + "Uncommited changes! Please commit before training.")
            exit()

        wandb.init(project="SSSLSquared", config=config)
        wandb.config["checkpoint_name"] = checkpoint_name
        wandb.config["train_transform"] = A.to_dict(train_transform)
        wandb.config["validation_transform"] = A.to_dict(val_transforms)

    model = SPLSS(in_channels=1, out_channels=config['num_classes']).to(DEVICE)
    CELoss = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = LRscheduler.PolynomialLR(optimizer, config['num_epochs'])

    train_ds = JonathanDataset(base_path=config['dataset_path'], transform=train_transform, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], num_workers=2, pin_memory=True, shuffle=True)
    val_ds = JonathanDataset(base_path=config['dataset_path'], transform=val_transforms, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], num_workers=2, pin_memory=True, shuffle=True)
    vid_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=True, shuffle=False)

    if LOG_WANDB:
        wandb.watch(model)
    
    # Save config stuff
    A.save(train_transform, "checkpoints/" + checkpoint_name + "/train_transform.yaml", data_format="yaml")
    A.save(val_transforms, "checkpoints/" + checkpoint_name + "/val_transform.yaml", data_format="yaml")

    with open("checkpoints/" + checkpoint_name + "/config.yml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    for epoch in range(config['num_epochs']):
        evaluate(val_loader, model, CELoss, epoch, log_wandb=LOG_WANDB)
        visualize(val_loader, model, epoch, log_wandb=LOG_WANDB)
        train(train_loader, CELoss, model, optimizer, epoch, log_wandb=LOG_WANDB)

        checkpoint = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()} | model.get_statedict()
        torch.save(checkpoint, "checkpoints/" + checkpoint_name + "/model.pth.tar")
    
    generate_video(model, vid_loader, "checkpoints/" + checkpoint_name + "/", log_wandb=LOG_WANDB)
    print("\033[92m" + "Training Done!")


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def train(train_loader, loss_func, model, optim, epoch, log_wandb = False):
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

    if log_wandb:
        print("Logging wandb")
        wandb.log({"Loss": running_average / len(train_loader)}, step=epoch)


def visualize(val_loader, model, epoch, log_wandb = False):
    if not log_wandb:
        return

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


def evaluate(val_loader, model, loss_func, epoch, log_wandb = False):
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
    
    if log_wandb:
        wandb.log({"Eval Loss": running_average / len(val_loader)}, step=epoch)
        wandb.log({"Eval Accuracy": total_acc}, step=epoch)
        wandb.log({"Eval DICE": total_dice}, step=epoch)

    print("Eval Loss at Epoch {0}: {1}".format(epoch, running_average / len(val_loader)))
    print("Eval Accuracy at Epoch {0}: {1}".format(epoch, total_acc))
    print("Eval DICE at Epoch {0}: {1}".format(epoch, total_dice))

def generate_video(model, data_loader, path, num_frames = 100, log_wandb = False):
    model.eval()
    count = 0
    video_list = []
    
    for images, gt_seg in data_loader:
        if count > num_frames:
            break

        images = images.to(device=DEVICE)
        gt_seg = gt_seg.to(device=DEVICE)

        pred_seg = model(images).softmax(dim=1).argmax(dim=1)

        visualizer = Visualizer.Visualize2D(x=1, y=1, remove_border=True)
        visualizer.draw_images(images)
        visualizer.draw_segmentation(pred_seg, 4, opacity=0.8)

        frame = visualizer.get_as_numpy_arr()
        
        # CHANNELS x WIDTH x HEIGHT!!!
        frame = frame.reshape(frame.shape[2], frame.shape[1], frame.shape[0])
        video_list.append(frame)
        count += images.shape[0]

    video = np.stack(video_list, axis=0)

    if log_wandb:
        wandb.log({"video": wandb.Video(video, fps=4)})

    out = cv2.VideoWriter(path + "val_video.mp4",cv2.VideoWriter_fourcc(*'mp4v'), 10, (video_list[0].shape[1], video_list[0].shape[2]))
    for frame in video_list:
        out.write(frame.reshape(frame.shape[2], frame.shape[1], frame.shape[0]))
    out.release()

if __name__ == "__main__":
    main()