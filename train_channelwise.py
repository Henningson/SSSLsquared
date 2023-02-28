import torch
import albumentations as A
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import LRscheduler
import datetime
import yaml
from evaluate import evaluate
from printer import Printer
import os
import argparse
import pygit2
import Visualizer
import cv2
import utils
import Losses
import ConfigArgsParser
from models.LSQ import LSQLocalization
import sys
import random
import Args
import utils
import wandb

import kornia
sys.path.append("models/")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def main():
    parser = Args.GlobalArgumentParser(
                    prog = 'Keypoint Regularized Training for Semantic Segmentation',
                    description = 'Train a Segmentation Network that is optimized for simultaneously outputting keypoints',
                    epilog = 'Arguments can be used to overwrite values in a config file.')
    
    args = parser.parse_args()
    
    LOG_WANDB = args.logwandb
    LOAD_FROM_CHECKPOINT = args.checkpoint is not None

    if args.checkpoint_name:
        CHECKPOINT_PATH = os.path.join("checkpoints", args.checkpoint_name)
    else:
        CHECKPOINT_PATH = args.checkpoint if LOAD_FROM_CHECKPOINT else os.path.join("checkpoints", datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
    
    CHECKPOINT_NAME = CHECKPOINT_PATH.split("/")[-1]

    # Always add magic number to path ._.
    if not LOAD_FROM_CHECKPOINT:
        magic_number = str(random.randint(0, 10000))
        CHECKPOINT_NAME += "_" + magic_number
        CHECKPOINT_PATH += "_" + magic_number

    CONFIG_PATH = os.path.join(CHECKPOINT_PATH, "config.yml") if LOAD_FROM_CHECKPOINT else args.config
    TRAIN_TRANSFORM_PATH = os.path.join(CHECKPOINT_PATH, "train_transform.yaml") if LOAD_FROM_CHECKPOINT else "train_transform.yaml"
    VAL_TRANSFORM_PATH = os.path.join(CHECKPOINT_PATH, "val_transform.yaml") if LOAD_FROM_CHECKPOINT else "val_transform.yaml"

    config = ConfigArgsParser.ConfigArgsParser(utils.load_config(CONFIG_PATH), args)
    
    # Gotta check this manually as sweeps do not allow nested lists
    if args.model_depth is not None:
        if args.model_depth == 0:
            config["features"] = [64, 128, 256]
        elif args.model_depth == 1:
            config["features"] = [64, 128, 256, 512]
        elif args.model_depth == 2:
            config["features"] = [64, 128, 256, 512, 1024]
        elif args.model_depth == 3:
            config["features"] = [32, 64, 128, 256]
        elif args.model_depth == 4:
            config["features"] = [32, 64, 128, 256, 512]
        elif args.model_depth == 5:
            config["features"] = [32, 64, 128, 256, 512, 1024]

    if not LOAD_FROM_CHECKPOINT:
        os.mkdir(CHECKPOINT_PATH)

    train_transform = A.load(TRAIN_TRANSFORM_PATH, data_format='yaml')
    val_transforms = A.load(VAL_TRANSFORM_PATH, data_format='yaml')

    neuralNet = __import__(config["model"])
    model = neuralNet.Model(config=config, state_dict=torch.load(os.path.join("pretrained", str(config["features"]) + ".pth.tar")) if args.pretrain else None, pretrain=True).to(DEVICE)
    loss = nn.CrossEntropyLoss(weight=torch.tensor(config["loss_weights"], dtype=torch.float32, device=DEVICE))
    #loss = kornia.losses.dice_loss
    cpu_loss = nn.CrossEntropyLoss(weight=torch.tensor(config["loss_weights"], dtype=torch.float32, device="cpu"))

    if LOG_WANDB:
        repo = pygit2.Repository('.')
        num_uncommitted_files = repo.diff().stats.files_changed

        if num_uncommitted_files > 0:
            Printer.Warning("Uncommited changes! Please commit before training.")
            exit()

        wandb.init(project="SSSLSquared", config=config)
        wandb.config["loss"] = type(loss).__name__
        wandb.config["checkpoint_name"] = CHECKPOINT_NAME
        wandb.config["train_transform"] = A.to_dict(train_transform)
        wandb.config["validation_transform"] = A.to_dict(val_transforms)

    
    config.printDifferences(utils.load_config(CONFIG_PATH))

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=config['learning_rate']) if config['optimizer'] == "adam" else optim.SGD(parameters, lr=config['learning_rate'])

    scheduler = LRscheduler.PolynomialLR(optimizer, config['num_epochs'], last_epoch=config['last_epoch'])


    dataset = __import__('dataset').__dict__[config['dataset_name']]
    train_ds = dataset(config=config, is_train=True, transform=train_transform)
    val_ds = dataset(config=config, is_train=False, transform=val_transforms)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], num_workers=config['num_workers'], pin_memory=True, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], num_workers=config['num_workers'], pin_memory=True, shuffle=False)

    localizer = LSQLocalization(local_maxima_window = config["maxima_window"], 
                                    gauss_window = config["gauss_window"], 
                                    heatmapaxis = config["heatmapaxis"], 
                                    threshold = config["threshold"])

    if LOG_WANDB:
        wandb.watch(model)
    
    # Save config stuff
    A.save(train_transform, CHECKPOINT_PATH + "/train_transform.yaml", data_format="yaml")
    A.save(val_transforms, CHECKPOINT_PATH + "/val_transform.yaml", data_format="yaml")

    for epoch in range(config['last_epoch'], config['num_epochs']):
        # Evaluate on Validation Set
        evaluate(val_loader, model, cpu_loss, localizer=localizer if epoch > config['keypoint_regularization_at'] else None, epoch=epoch, log_wandb=LOG_WANDB)

        # Visualize Validation as well as Training Set examples
        visualize(val_loader, model, epoch, title="Val Predictions", log_wandb=LOG_WANDB)
        visualize(train_loader, model, epoch, title="Train Predictions", log_wandb=LOG_WANDB)

        # Train the network
        train(train_loader, 
                loss, 
                model, 
                scheduler,
                epoch, 
                localizer, 
                use_regression=epoch > config['keypoint_regularization_at'] - 1,
                keypoint_lambda=config['keypoint_lambda'], 
                log_wandb=False,
                temporal_smoothing=epoch > config['temporal_regularization_at'] - 1,
                temporal_lambda=config['temporal_lambda'])

        checkpoint = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()} | model.get_statedict()
        torch.save(checkpoint, CHECKPOINT_PATH + "/model.pth.tar")

        config["last_epoch"] = epoch
        with open(CHECKPOINT_PATH + "/config.yml", 'w') as outfile:
            yaml.dump(dict(config), outfile, default_flow_style=False)

    Printer.OKG("Training Done!")

def train(train_loader, loss_func, model, scheduler, epoch, localizer, use_regression = False, keypoint_lambda=0.1, log_wandb = False, temporal_smoothing = False, heatmapaxis=3, temporal_lambda=0.01):
    Printer.Header("EPOCH: {0}".format(epoch))
    model.train()
    running_average = 0.0
    loop = tqdm(train_loader, desc="TRAINING")
    show = True
    for images, gt_seg, gt_keypoints in loop:
        scheduler.zero_grad()

        images = images.to(device=DEVICE)
        gt_seg = gt_seg.to(device=DEVICE)
        gt_keypoints = gt_keypoints.to(device=DEVICE)

        # forward
        pred_seg = model(images)
        
        loss = loss_func(pred_seg.float(), gt_seg.long())

        segmentation = pred_seg.softmax(dim=1)
        segmentation_argmax = segmentation.argmax(dim=1)
 
        if use_regression:
            try:
                _, pred_keypoints, _ = localizer.estimate(segmentation, torch.bitwise_or(segmentation_argmax == 2, segmentation_argmax == 3))
            except:
                Printer.Warning("Matrix probably singular. Whoopsie.")
                continue

            if pred_keypoints is not None:
                keypoint_loss = Losses.chamfer(pred_keypoints, gt_keypoints)
                loss += keypoint_lambda * (keypoint_loss if type(keypoint_loss) == float else keypoint_loss.item())

        if temporal_smoothing:
            loss += temporal_lambda * segmentation[:, heatmapaxis, :, :].var(dim=0).mean()


        loss.backward()
        scheduler.step()

        running_average += loss.item()
        loop.set_postfix(loss=loss.item())
    scheduler.update_lr()

    if log_wandb:
        print("Logging wandb")
        wandb.log({"Loss": running_average / len(train_loader)}, step=epoch)


def visualize(val_loader, model, epoch, title="Validation Predictions", num_log=1, log_wandb=False):
    if not log_wandb:
        return

    model.eval()
    for images, gt_seg, _ in val_loader:
        images = images.to(device=DEVICE)
        gt_seg = gt_seg.to(device=DEVICE)

        pred_seg = model(images).softmax(dim=1).argmax(dim=1)
        images = utils.normalize_image_batch(images)


        for i in range(num_log):
            wandb.log(
            {"{0} {1}".format(title, i) : wandb.Image(images[i].detach().cpu().moveaxis(0, -1).numpy(), masks={
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