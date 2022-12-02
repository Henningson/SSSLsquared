import torch
import torch.nn as nn
import albumentations as A
import argparse
import os
import utils
import numpy as np
import matplotlib.cm as cm
import viewer
import torchmetrics
import metrics
import wandb
from darktheme.widget_template import DarkPalette

from PyQt5.QtWidgets import QApplication
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import HLEPlusPlus
from models.LSQ import LSQLocalization

import sys
sys.path.append("models/")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(val_loader, model, loss_func, localizer=None, epoch = -1, log_wandb = False):
    Accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=4)
    DICE = torchmetrics.Dice(num_classes=4)
    IOU = torchmetrics.JaccardIndex(num_classes=4)

    nnMSE = metrics.nnMSE(threshold=2.0)
    nnPrecision = metrics.nnPrecision(threshold=2.0)

    running_average = 0.0
    inference_time = 0
    point_detection_time = 0
    num_images = 0

    model.eval()
    loop = tqdm(val_loader, desc="EVAL")
    for images, gt_seg, keypoints in loop:
        images = images.to(device=DEVICE)
        gt_seg = gt_seg.long()

        torch.cuda.synchronize()

        starter_cnn, ender_cnn = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
        starter_cnn.record()
        pred_seg = model(images)
        torch.cuda.synchronize()
        ender_cnn.record()

        acc = Accuracy(pred_seg.softmax(dim=1).detach().cpu(), gt_seg)
        dice = DICE(pred_seg.softmax(dim=1).argmax(dim=1).detach().cpu(), gt_seg)
        iou = IOU(pred_seg.softmax(dim=1).argmax(dim=1).detach().cpu(), gt_seg)
        loss = loss_func(pred_seg.detach().cpu(), gt_seg).item()

        curr_time_cnn = starter_cnn.elapsed_time(ender_cnn)
        inference_time += curr_time_cnn
        num_images += images.shape[0]
        running_average += loss
        
        if localizer is not None:
            starter_lsq, ender_lsq = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
            starter_lsq.record()
            segmentation = pred_seg.softmax(dim=1)
            segmentation_argmax = segmentation.argmax(dim=1)
            _, pred_keypoints, _ = localizer.estimate(segmentation, torch.bitwise_or(segmentation_argmax == 2, segmentation_argmax == 3))
            torch.cuda.synchronize()
            ender_lsq.record()

            mse = nnMSE([i[:, [1, 0]].detach().cpu() for i in pred_keypoints], keypoints.detach().cpu())
            precision = nnPrecision([i[:, [1, 0]].detach().cpu() for i in pred_keypoints], keypoints.detach().cpu())

            curr_time_point_detection = starter_lsq.elapsed_time(ender_lsq)
            point_detection_time += curr_time_point_detection

        if localizer is not None:
            loop.set_postfix({"DICE": dice, "ACC": acc, "Loss": loss, "IOU": iou, "Precision": precision, "MSE": mse, "Infer. Time": curr_time_cnn, "Point Pred. Time:": curr_time_point_detection})
        else:
            loop.set_postfix({"DICE": dice, "ACC": acc, "Loss": loss, "IOU": iou, "Infer. Time": curr_time_cnn})

    total_acc = Accuracy.compute()
    total_dice = DICE.compute()
    total_IOU = IOU.compute()
    eval_loss = running_average / len(val_loader)

    if localizer is not None:
        total_mse = nnMSE.compute()
        total_precision = nnPrecision.compute()
    
    
    if log_wandb:
        wandb.log({"Eval Loss": eval_loss}, step=epoch)
        wandb.log({"Eval Accuracy": total_acc}, step=epoch)
        wandb.log({"Eval DICE": total_dice}, step=epoch)
        wandb.log({"Eval IOU": total_IOU}, step=epoch)
        wandb.log({"Inference Time (ms)": inference_time / num_images}, step=epoch)

    print("_____EPOCH {0}_____".format(epoch))
    print("Eval Loss: {1}".format(epoch, eval_loss))
    print("Eval Accuracy: {1}".format(epoch, total_acc))
    print("Eval IOU: {1}".format(epoch, total_IOU))
    print("Eval DICE {0}: {1}".format(epoch, total_dice))
    print("Inference Speed (ms): {:.3f}".format(inference_time / num_images))
    
    if localizer is not None:
        print("Eval MSE: {1}".format(epoch, total_mse))
        print("Eval PRECISION {0}: {1}".format(epoch, total_precision))
        print("Point Detection (ms): {:.3f}".format(point_detection_time / num_images))

    return eval_loss


def main():
    parser = argparse.ArgumentParser(
                    prog = 'Evaluation for trained Deep Neural Networks',
                    description = 'Loads  as input, and evaluate it based on the keys given in the config file.',
                    epilog = 'For questions, generate an issue at: https://github.com/Henningson/SSSLsquared or write an E-Mail to: jann-ole.henningson@fau.de')
    parser.add_argument("-c", "--checkpoint", type=str, default="checkpoints/2022-12-01-10:52:50/")
    
    args = parser.parse_args()
    checkpoint_path = args.checkpoint

    if checkpoint_path == "" or not os.path.isdir(checkpoint_path):
        print("\033[93m" + "Please provide a viable checkpoint path")

    config = utils.load_config(os.path.join(checkpoint_path, "config.yml"))
    val_transforms = A.load(os.path.join(checkpoint_path, "val_transform.yaml"), data_format='yaml')

    neuralNet = __import__(config["model"])
    model = neuralNet.Model(in_channels=1, 
                            out_channels=config['num_classes'], 
                            features=config['features'], 
                            state_dict=torch.load(os.path.join(checkpoint_path, "model.pth.tar"))
                            ).to(DEVICE)
    loss = nn.CrossEntropyLoss()

    val_ds = HLEPlusPlus(base_path=config['dataset_path'], 
                         keys=config['val_keys'].split(","), 
                         transform=val_transforms)

    val_loader = DataLoader(val_ds, 
                            batch_size=config['batch_size'], 
                            num_workers=4, 
                            pin_memory=True, 
                            shuffle=False)
    
    localizer = LSQLocalization(local_maxima_window = config["maxima_window"], 
                                gauss_window = config["gauss_window"], 
                                heatmapaxis = config["heatmapaxis"], 
                                threshold = config["threshold"])

    evaluate(val_loader, model, loss, localizer=localizer)

if __name__ == "__main__":
    main()