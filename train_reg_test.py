import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset import HLEPlusPlus
from torch.utils.data import DataLoader
import LRscheduler
import datetime
import yaml
from pathlib import Path
from evaluate import evaluate
import os
import argparse
import pygit2
import Visualizer
import numpy as np
import cv2
import utils
import Losses
import evaluate
import gc
import train_reg
from models.LSQ import LSQLocalization

import sys
sys.path.append("models/")



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import wandb

def main():
    parser = argparse.ArgumentParser(
                    prog = 'Inference for Deep Neural Networks',
                    description = 'Loads  as input, and visualize it based on the keys given in the config file.',
                    epilog = 'For question, generate an issue at: https://github.com/Henningson/SSSLsquared or write an E-Mail to: jann-ole.henningson@fau.de')
    parser.add_argument("-c", "--checkpoint", type=str, default="checkpoints/2022-12-06-13:34:51/")
    
    args = parser.parse_args()
    checkpoint_path = args.checkpoint

    if checkpoint_path == "" or not os.path.isdir(checkpoint_path):
        print("\033[93m" + "Please provide a viable checkpoint path")

    config = utils.load_config(os.path.join(checkpoint_path, "config.yml"))
    train_transform = A.load(os.path.join(checkpoint_path, "train_transform.yaml"), data_format='yaml')
    val_transforms = A.load(os.path.join(checkpoint_path, "val_transform.yaml"), data_format='yaml')

    neuralNet = __import__(config["model"])
    model = neuralNet.Model(in_channels=1, 
                            out_channels=config['num_classes'], 
                            features=config['features'], 
                            state_dict=torch.load(os.path.join(checkpoint_path, "model.pth.tar"))
                            ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = LRscheduler.PolynomialLR(optimizer, config['num_epochs'], last_epoch=config['last_epoch'])

    train_ds = HLEPlusPlus(base_path=config['dataset_path'], keys=config['train_keys'].split(","), pad_keypoints=config['pad_keypoints'], transform=train_transform)
    val_ds = HLEPlusPlus(base_path=config['dataset_path'], keys=config['val_keys'].split(","), pad_keypoints=config['pad_keypoints'], transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], num_workers=2, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], num_workers=2, pin_memory=True, shuffle=True)
    loss = nn.CrossEntropyLoss()
    for epoch in range(config['num_epochs']):
        localizer = LSQLocalization(local_maxima_window = config["maxima_window"], 
                                gauss_window = config["gauss_window"], 
                                heatmapaxis = config["heatmapaxis"], 
                                threshold = config["threshold"])

        evaluate.evaluate(val_loader, model, loss, localizer=localizer, epoch=epoch)

        train_reg.train(train_loader, 
                loss, 
                model, 
                scheduler, 
                epoch, 
                localizer, 
                start_reg=epoch > config['keypoint_regularization_at'],
                keypoint_lambda=config['keypoint_lambda'], 
                log_wandb=False)

        checkpoint = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()} | model.get_statedict()
        torch.save(checkpoint, os.path.join(checkpoint_path, "model.pth.tar"))

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