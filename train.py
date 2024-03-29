import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from dataset import HLEPlusPlus
from torch.utils.data import DataLoader
import datetime
import yaml
from evaluate import evaluate
import os
import argparse
import pygit2
import numpy as np
import cv2

import Utils.LRscheduler as LRscheduler
import Utils.utils as utils
import Visualization.Matplot as Visualizer

import sys
sys.path.append("models/")



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

    config = utils.load_config("config.yml")
    checkpoint_name = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    os.mkdir("checkpoints/" + checkpoint_name)

    train_transform = A.Compose(
        [
            A.Resize(height=config['image_height'], width=config['image_width']),
            A.Affine(translate_percent = 0.15, p=0.5),
            A.Rotate(limit=40, border_mode = cv2.BORDER_CONSTANT, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Perspective(scale=(0.05, 0.2), p=0.5),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format='xy')
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
        keypoint_params=A.KeypointParams(format='xy')
    )

    neuralNet = __import__(config["model"])
    model = neuralNet.Model(in_channels=1, out_channels=config['num_classes'], features=config['features']).to(DEVICE)
    loss = nn.CrossEntropyLoss()

    if LOG_WANDB:
        repo = pygit2.Repository('.')
        num_uncommitted_files = repo.diff().stats.files_changed

        if num_uncommitted_files > 0:
            print("\033[93m" + "Uncommited changes! Please commit before training.")
            exit()

        wandb.init(project="SSSLSquared", config=config)
        wandb.config["loss"] = type(loss).__name__
        wandb.config["checkpoint_name"] = checkpoint_name
        wandb.config["train_transform"] = A.to_dict(train_transform)
        wandb.config["validation_transform"] = A.to_dict(val_transforms)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = LRscheduler.PolynomialLR(optimizer, config['num_epochs'])

    train_ds = HLEPlusPlus(base_path=config['dataset_path'], keys=config['train_keys'].split(","), pad_keypoints=config['pad_keypoints'], transform=train_transform)
    val_ds = HLEPlusPlus(base_path=config['dataset_path'], keys=config['val_keys'].split(","), pad_keypoints=config['pad_keypoints'], transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], num_workers=2, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], num_workers=2, pin_memory=True, shuffle=True)

    vid_loader_val = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=True, shuffle=False)
    vid_loader_train = DataLoader(train_ds, batch_size=1, num_workers=2, pin_memory=True, shuffle=False)

    if LOG_WANDB:
        wandb.watch(model)
        wandb.config["dataset_name"] = type(train_ds).__name__
    
    # Save config stuff
    A.save(train_transform, "checkpoints/" + checkpoint_name + "/train_transform.yaml", data_format="yaml")
    A.save(val_transforms, "checkpoints/" + checkpoint_name + "/val_transform.yaml", data_format="yaml")

    with open("checkpoints/" + checkpoint_name + "/config.yml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    for epoch in range(config['num_epochs']):
        # Evaluate on Validation Set
        eval_loss = evaluate(val_loader, model, loss, epoch=epoch, log_wandb=LOG_WANDB)

        #if eval_loss < config['eval_threshold']:
        #   break

        # Visualize Validation as well as Training Set examples
        visualize(val_loader, model, epoch, title="Val Predictions", log_wandb=LOG_WANDB)
        visualize(train_loader, model, epoch, title="Train Predictions", log_wandb=LOG_WANDB)

        # Train the network
        train(train_loader, loss, model, optimizer, epoch, log_wandb=LOG_WANDB)

        checkpoint = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()} | model.get_statedict()
        torch.save(checkpoint, "checkpoints/" + checkpoint_name + "/model.pth.tar")
    
    generate_video(model, vid_loader_val, "checkpoints/" + checkpoint_name + "/val_video.mp4")
    generate_video(model, vid_loader_train, "checkpoints/" + checkpoint_name + "/train_video.mp4")
    print("\033[92m" + "Training Done!")


def train(train_loader, loss_func, model, optim, epoch, log_wandb = False):
    model.train()
    running_average = 0.0
    loop = tqdm(train_loader, desc="TRAINING")
    for images, gt_seg, _ in loop:
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


def visualize(val_loader, model, epoch, title="Validation Predictions", num_log=2, log_wandb=False):
    if not log_wandb:
        return

    model.eval()
    for images, gt_seg, _ in val_loader:
        images = images.to(device=DEVICE)
        gt_seg = gt_seg.to(device=DEVICE)

        pred_seg = model(images).softmax(dim=1).argmax(dim=1)

        for i in range(num_log):
            wandb.log(
            {"{0} {1}".format(title, i) : wandb.Image(images[i].detach().cpu().numpy(), masks={
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


def generate_video(model, data_loader, path):
    model.eval()
    count = 0
    video_list = []
    
    for images, gt_seg, _ in data_loader:
        images = images.to(device=DEVICE)
        gt_seg = gt_seg.to(device=DEVICE)

        pred_seg = model(images).softmax(dim=1).argmax(dim=1)

        visualizer = Visualizer.Visualize2D(x=1, y=1, remove_border=True, do_not_open=True)
        visualizer.draw_images(images)
        visualizer.draw_segmentation(pred_seg, 4, opacity=0.8)

        frame = visualizer.get_as_numpy_arr()
        visualizer.close()

        # CHANNELS x WIDTH x HEIGHT!!!
        frame = frame.reshape(frame.shape[2], frame.shape[1], frame.shape[0])
        video_list.append(frame)
        count += images.shape[0]

    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (video_list[0].shape[1], video_list[0].shape[2]))
    for frame in video_list:
        out.write(frame.reshape(frame.shape[2], frame.shape[1], frame.shape[0]))
    out.release()

if __name__ == "__main__":
    main()