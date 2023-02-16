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
import metrics_dom

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

    running_average = 0.0
    inference_time = 0
    point_detection_time = 0
    num_images = 0

    model.eval()
    
    TP = 0
    FP = 0
    FN = 0
    inference_time = 0
    loop = tqdm(val_loader, desc="EVAL")
    for images, gt_seg, keypoints in loop:
        images = images.to(device=DEVICE)
        gt_seg = gt_seg.long()
        
        gt_keypoints = keypoints.split(1, dim=0)
        gt_keypoints = [keys[0][~torch.isnan(keys[0]).any(axis=1)][:, [1, 0]] for keys in gt_keypoints]

        torch.cuda.synchronize()

        starter_cnn, ender_cnn = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
        starter_cnn.record()
        pred_seg = model(images)
        torch.cuda.synchronize()
        ender_cnn.record()

        acc = Accuracy(pred_seg.softmax(dim=1).detach().cpu(), gt_seg)
        dice = DICE(pred_seg.softmax(dim=1).argmax(dim=1).detach().cpu(), gt_seg)
        iou = IOU(pred_seg.softmax(dim=1).argmax(dim=1).detach().cpu(), gt_seg)
        loss = loss_func.cpu()(pred_seg.detach().cpu(), gt_seg).item()

        curr_time_cnn = starter_cnn.elapsed_time(ender_cnn)
        inference_time += curr_time_cnn
        num_images += images.shape[0]
        running_average += loss
        
        if localizer is not None:
            starter_lsq, ender_lsq = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
            starter_lsq.record()
            segmentation = pred_seg.softmax(dim=1)
            segmentation_argmax = segmentation.argmax(dim=1)
            try:
                _, pred_keypoints, _ = localizer.estimate(segmentation, torch.bitwise_or(segmentation_argmax == 2, segmentation_argmax == 3))
            except:
                print("Matrix probably singular. Whoopsie.")
                continue
            
            torch.cuda.synchronize()
            ender_lsq.record()

            if pred_keypoints is None:
                continue
            
            
            TP_temp, FP_temp, FN_temp = metrics_dom.keypoint_statistics(pred_keypoints, gt_keypoints, 2.0, prediction_format="yx", target_format="yx")
            TP += TP_temp
            FP += FP_temp
            FN += FN_temp

            curr_time_point_detection = starter_lsq.elapsed_time(ender_lsq)
            point_detection_time += curr_time_point_detection

        if localizer is not None:
            loop.set_postfix({"DICE": dice, "ACC": acc, "Loss": loss, "IOU": iou, "Precision": precision, "MSE": mse, "Infer. Time": curr_time_cnn, "Point Pred. Time:": curr_time_point_detection})
        else:
            loop.set_postfix({"DICE": dice, "ACC": acc, "Loss": loss, "IOU": iou, "Infer. Time": curr_time_cnn})

    # Segmentation
    total_acc = Accuracy.compute()
    total_dice = DICE.compute()
    total_IOU = IOU.compute()
    eval_loss = running_average / len(val_loader)



    if localizer is not None:
        # Keypoint Stuff
        ap = metrics_dom.average_precision(TP, FP, FN)
        f1 = metrics_dom.f1_score(TP, FP, FN)
        keypoint_dice = metrics_dom.dice_score(TP, FP, FN)
        recoll = metrics_dom.recall(TP, FP, FN)
        prec = metrics_dom.precision(TP, FP, FN)
    
    
    if log_wandb:
        wandb.log({"Eval Loss": eval_loss}, step=epoch)
        wandb.log({"Eval Accuracy": total_acc}, step=epoch)
        wandb.log({"Eval DICE": total_dice}, step=epoch)
        wandb.log({"Eval IOU": total_IOU}, step=epoch)
        wandb.log({"Inference Time (ms)": inference_time / num_images}, step=epoch)


    print("_______SEGMENTATION STUFF_______")
    print("Eval Loss: {1}".format(epoch, eval_loss))
    print("Eval Accuracy: {1}".format(epoch, total_acc))
    print("Eval IOU: {1}".format(epoch, total_IOU))
    print("Eval DICE {0}: {1}".format(epoch, total_dice))
    print("Inference Speed (ms): {:.4f}".format(inference_time / num_images))
    
    if localizer is not None:
        print("_______KEYPOINT STUFF_______")
        print("AP: {:.4f}".format(ap))
        print("F1: {:.4f}".format(f1))
        print("DICE: {:.4f}".format(keypoint_dice))
        print("Recall: {:.4f}".format(recoll))
        print("Precision: {:.4f}".format(prec))
        print("Complete Time(ms): {:.4f}".format((inference_time + point_detection_time) / num_images))

    return eval_loss


def main():

    parser = argparse.ArgumentParser(
                    prog = 'Inference for Deep Neural Networks',
                    description = 'Loads  as input, and visualize it based on the keys given in the config file.',
                    epilog = 'For question, generate an issue at: https://github.com/Henningson/SSSLsquared or write an E-Mail to: jann-ole.henningson@fau.de')
    parser.add_argument("-c", "--checkpoint", type=str, default="checkpoints/2023-01-31-17:14:43_9554/")
    parser.add_argument("-d", "--dataset_path", type=str, default='../HLEDataset/dataset/')

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    loss = torch.nn.CrossEntropyLoss()

    if checkpoint_path == "" or not os.path.isdir(checkpoint_path):
        print("\033[93m" + "Please provide a viable checkpoint path")

    config = utils.load_config(os.path.join(checkpoint_path, "config.yml"))
    config['dataset_path'] = args.dataset_path

    val_transforms = A.load(os.path.join(checkpoint_path, "val_transform.yaml"), data_format='yaml')

    neuralNet = __import__(config["model"])
    model = neuralNet.Model(config, state_dict=torch.load(os.path.join(checkpoint_path, "model.pth.tar"))).to(DEVICE)
    dataset = __import__('dataset').__dict__[config['dataset_name']]
    val_ds = dataset(config, is_train=False, transform=val_transforms)

    val_loader = DataLoader(val_ds, 
                            batch_size=config['batch_size'],
                            num_workers=2, 
                            pin_memory=True, 
                            shuffle=False)
    
    localizer = LSQLocalization(local_maxima_window = config["maxima_window"], 
                                gauss_window = config["gauss_window"], 
                                heatmapaxis = config["heatmapaxis"], 
                                threshold = 0.5,
                                device=DEVICE)

    evaluate(val_loader, model, loss, localizer=localizer)

if __name__ == "__main__":
    main()