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
import math
from torchmetrics.functional import dice, jaccard_index
import wandb
import metrics_dom
from typing import List, Union, Tuple, Optional
from chamferdist import ChamferDistance

from PyQt5.QtWidgets import QApplication
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import HLEPlusPlus
from models.LSQ import LSQLocalization

import sys
sys.path.append("models/")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(val_loader, model, loss_func, localizer=None, epoch = -1, log_wandb = False) -> Tuple[float, float, float, float, float, float, float]:
    running_average = 0.0
    inference_time = 0
    point_detection_time = 0
    num_images = 0
    num_images_for_inference_time = 0
    count = 0

    model.eval()

    dice_val = 0.0
    iou = 0.0
    cham = 0.0
    f1=0.0
    TP = 0
    FP = 0
    FN = 0

    chamloss = ChamferDistance()

    l2_distances  = []
    nme = 0.0
    precision = 0.0
    inference_time = 0
    loop = tqdm(val_loader, desc="EVAL")
    for images, gt_seg, keypoints in loop:
        count += 1

        images = images.to(device=DEVICE)
        gt_seg = gt_seg.long()
        
        keypoints = keypoints.float()
        gt_keypoints = keypoints.split(1, dim=0)
        gt_keypoints = [keys[0][~torch.isnan(keys[0]).any(axis=1)][:, [1, 0]] for keys in gt_keypoints]

        torch.cuda.synchronize()

        starter_cnn, ender_cnn = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
        starter_cnn.record()
        pred_seg = model(images)
        ender_cnn.record()
        torch.cuda.synchronize()

        softmax = pred_seg.softmax(dim=1).detach().cpu()
        argmax = softmax.argmax(dim=1)
        
        curr_time_cnn = starter_cnn.elapsed_time(ender_cnn)
        num_images += images.shape[0]

        if count > 2:
            num_images_for_inference_time += pred_seg.shape[0]
            inference_time += curr_time_cnn

        dice_val += dice(argmax, gt_seg, num_classes=4)
        iou += jaccard_index(argmax, gt_seg, num_classes=4)
        
        loss = loss_func.cpu()(pred_seg.detach().cpu(), gt_seg).item()
        running_average += loss

        
        if localizer is not None:
            segmentation = pred_seg.softmax(dim=1)
            segmentation_argmax = segmentation.argmax(dim=1)
            starter_lsq, ender_lsq = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
            starter_lsq.record()
            try:
                _, pred_keypoints, _ = localizer.estimate(segmentation)
            except:
                print("Matrix probably singular. Whoopsie.")
                continue
            ender_lsq.record()
            torch.cuda.synchronize()

            if pred_keypoints is None:
                continue
            
            
            TP_temp, FP_temp, FN_temp, distances = metrics_dom.keypoint_statistics(pred_keypoints, gt_keypoints, 2.0, prediction_format="yx", target_format="yx")
            TP += TP_temp
            FP += FP_temp
            FN += FN_temp
            l2_distances = l2_distances + distances

            for i in range(len(pred_keypoints)):
                cham += (chamloss(gt_keypoints[i].unsqueeze(0), pred_keypoints[i].unsqueeze(0).detach().cpu(), bidirectional=True) / math.sqrt(512*512 + 256*256))

            curr_time_point_detection = starter_lsq.elapsed_time(ender_lsq)
            if count > 2:
                point_detection_time += curr_time_point_detection

        if localizer is not None:
            loop.set_postfix({"DICE": dice, "Loss": loss, "IOU": iou, "Infer. Time": curr_time_cnn, "Point Pred. Time:": curr_time_point_detection})
        else:
            loop.set_postfix({"DICE": dice, "Loss": loss, "IOU": iou, "Infer. Time": curr_time_cnn})
        
        count += 1

    # Segmentation
    total_dice = dice_val / num_images
    total_IOU = iou / num_images
    total_CHAM = cham / num_images
    eval_loss = running_average / len(val_loader)



    if localizer is not None:
        # Keypoint Stuff
        try:
            precision = metrics_dom.precision(TP, FP, FN)
            f1 = metrics_dom.f1_score(TP, FP, FN)
            nme = sum(l2_distances)/len(l2_distances)
        except:
            precision = 0.0
            ap = 0.0
            f1 = 0.0
            nme = 0.0
    
    
    if log_wandb:
        wandb.log({"Eval Loss": eval_loss}, step=epoch)
        wandb.log({"Eval DICE": total_dice}, step=epoch)
        wandb.log({"Eval IOU": total_IOU}, step=epoch)
        wandb.log({"Inference Time (ms)": inference_time / num_images}, step=epoch)

    print("_______SEGMENTATION STUFF_______")
    print("Eval Loss: {1}".format(epoch, eval_loss))
    print("Eval IOU: {1}".format(epoch, total_IOU))
    print("Eval DICE {0}: {1}".format(epoch, total_dice))
    
    if localizer is not None:
        print("_______KEYPOINT STUFF_______")
        print("Precision: {0}".format(float(precision)))
        print("F1: {0}".format(float(f1)))
        print("NME: {0}".format(float(nme)))
        print("ChamferDistance: {0}".format(float(total_CHAM)))

        
        print("Inference Speed (ms): {0}".format(inference_time / num_images))
        print("Point Speed (ms): {0}".format(point_detection_time / num_images))
        print("Complete Time(ms): {0}".format((inference_time + point_detection_time) / num_images))

    return float(precision), float(f1), float(nme), float(total_IOU), float(total_dice), float((inference_time + point_detection_time) / num_images), float(total_CHAM)


class BaseMetric:
    def __init__(self, is_keypoint_metric = False):
        self.is_keypoint_metric = is_keypoint_metric
        self.running_average = 0.0
        self.num_datapoints = 0

    def isKeypointMetric(self):
        return self.is_keypoint_metric

    def compute(self, prediction, target) -> float:
        # TODO: Implement me
        return None

    def get_final_score(self):
        return self.running_average / self.num_datapoints


class DiceMetric(BaseMetric):
    def __init__(self, num_classes=4):
        super().__init__(is_keypoint_metric = False)
        self.num_classes = num_classes
        self.name = "DICE"

    def compute(self, prediction, target) -> float:
        score = dice(prediction, target, num_classes=self.num_classes, ignore_index=0)
        self.running_average += score
        self.num_datapoints += 1
        return score


class JaccardIndexMetric(BaseMetric):
    def __init__(self, num_classes=4):
        super().__init__(is_keypoint_metric = False)
        self.num_classes = num_classes
        self.name = "IoU"

    def compute(self, prediction, target) -> float:
        score = jaccard_index(prediction, target, num_classes=self.num_classes, ignore_index=0)
        self.running_average += score
        self.num_datapoints += 1
        return score
    

class F1ScoreMetric(BaseMetric):
    def __init__(self, outlier_threshold=2.0, prediction_format="yx", target_format="yx"):
        super().__init__(is_keypoint_metric = True)
        self.outlier_threshold = outlier_threshold
        self.prediction_format = prediction_format
        self.target_format = target_format
        self.TOTAL_TP = 0
        self.TOTAL_FP = 0
        self.TOTAL_TN = 0
        self.TOTAL_FN = 0
        self.name = "F1-Score"

    def compute(self, prediction, target) -> float:
        TP, FP, FN, _ = metrics_dom.keypoint_statistics(prediction, target, 
                                                        self.outlier_threshold, 
                                                        prediction_format=self.prediction_format, 
                                                        target_format=self.target_format)
        self.TOTAL_TP += TP
        self.TOTAL_FP += FP
        self.TOTAL_FN += FN
        self.num_datapoints += 1
        return metrics_dom.f1_score(TP, FP, FN)


    def get_final_score(self):
        return metrics_dom.f1_score(self.TOTAL_TP, self.TOTAL_FP, self.TOTAL_FN)



class AveragePrecisionMetric(BaseMetric):
    def __init__(self, outlier_threshold=2.0, prediction_format="yx", target_format="yx"):
        super().__init__(is_keypoint_metric = True)
        self.outlier_threshold = outlier_threshold
        self.prediction_format = prediction_format
        self.target_format = target_format
        self.TOTAL_TP = 0
        self.TOTAL_FP = 0
        self.TOTAL_TN = 0
        self.TOTAL_FN = 0
        self.name = "Precision"

    def compute(self, prediction, target) -> float:
        TP, FP, FN, _ = metrics_dom.keypoint_statistics(prediction, target, 
                                                        self.outlier_threshold, 
                                                        prediction_format=self.prediction_format, 
                                                        target_format=self.target_format)
        self.TOTAL_TP += TP
        self.TOTAL_FP += FP
        self.TOTAL_FN += FN
        self.num_datapoints += 1
        return metrics_dom.average_precision(TP, FP, FN)


    def get_final_score(self):
        return metrics_dom.average_precision(self.TOTAL_TP, self.TOTAL_FP, self.TOTAL_FN)
    

class PrecisionMetric(BaseMetric):
    def __init__(self, outlier_threshold=2.0, prediction_format="yx", target_format="yx"):
        super().__init__(is_keypoint_metric = True)
        self.outlier_threshold = outlier_threshold
        self.prediction_format = prediction_format
        self.target_format = target_format
        self.TOTAL_TP = 0
        self.TOTAL_FP = 0
        self.TOTAL_TN = 0
        self.TOTAL_FN = 0
        self.name = "Precision"

    def compute(self, prediction, target) -> float:
        TP, FP, FN, _ = metrics_dom.keypoint_statistics(prediction, target, 
                                                        self.outlier_threshold, 
                                                        prediction_format=self.prediction_format, 
                                                        target_format=self.target_format)
        self.TOTAL_TP += TP
        self.TOTAL_FP += FP
        self.TOTAL_FN += FN
        self.num_datapoints += 1
        return metrics_dom.precision(TP, FP, FN)


    def get_final_score(self):
        return metrics_dom.precision(self.TOTAL_TP, self.TOTAL_FP, self.TOTAL_FN)


class NMEMetric(BaseMetric):
    def __init__(self, outlier_threshold=2.0, prediction_format="yx", target_format="yx"):
        super().__init__(is_keypoint_metric = True)
        self.outlier_threshold = outlier_threshold
        self.prediction_format = prediction_format
        self.target_format = target_format
        self.total_l2_distances = []
        self.name = "NME"

    def compute(self, prediction, target) -> float:
        _, _, _, inlier_distances = metrics_dom.keypoint_statistics(prediction, target, 
                                                        self.outlier_threshold, 
                                                        prediction_format=self.prediction_format, 
                                                        target_format=self.target_format)
        
        self.total_l2_distances = self.total_l2_distances + inlier_distances

        return sum(inlier_distances) / len(inlier_distances)


    def get_final_score(self):
        return sum(self.total_l2_distances) / len(self.total_l2_distances)


class ChamferMetric(BaseMetric):
    def __init__(self):
        super().__init__(is_keypoint_metric = True)
        self.name = "Chamfer"
        self.chamf = ChamferDistance()

    def compute(self, prediction, target) -> float:
        count = 0 
        current_average = 0
        for i in range(len(prediction)):
            chamferdist = self.chamf(prediction[i].unsqueeze(0), target[i].unsqueeze(0), bidirectional=True) / torch.sqrt(torch.tensor(512*512 + 256*256))
            count += 1
            current_average += chamferdist 
            
            self.running_average += chamferdist
            self.num_datapoints += 1

        return current_average / count


class BaseEvaluator:
    def __init__(self, model, val_loader, localizer, config, metrics):
        self.model = model
        self.val_loader = val_loader
        self.config = config
        self.metrics = metrics
        self.localizer = localizer
        self.running_metrics = None

    def evaluate(self):
        loop = tqdm(self.val_loader, desc="Evaluation")
        for images, gt_seg, keypoints in loop:
            images = images.to(device=DEVICE)
            gt_seg = gt_seg.to(device=DEVICE).long()
            keypoints = keypoints.to(device=DEVICE)
            
            keypoints = keypoints.float().split(1, dim=0)
            keypoints = [keys[0][~torch.isnan(keys[0]).any(axis=1)][:, [1, 0]] for keys in keypoints]

            prediction = self.model(images).softmax(dim=1)
            segmentation = prediction.argmax(dim=1)
            _, means, _ = self.localizer.estimate(prediction, segmentation=torch.bitwise_or(segmentation == 2, segmentation == 3))

            metric_dict = {}
            for metric in self.metrics:
                if metric.isKeypointMetric():
                    metric_dict[metric.name] = metric.compute(means, keypoints)
                else:
                    metric_dict[metric.name] = metric.compute(segmentation, gt_seg)
            
            loop.set_postfix(metric_dict)

    def get_final_metrics(self):
        return [metric.get_final_score() for metric in self.metrics]


class Evaluator2D3D:
    def __init__(self, model, val_loader, localizer, config, metrics):
        self.model = model
        self.val_loader = val_loader
        self.config = config
        self.metrics = metrics
        self.localizer = localizer
        self.running_metrics = None


    def evaluate(self):
        loop = tqdm(self.val_loader, desc="Evaluation")
        for images, gt_seg, keypoints in loop:
            if images.shape[0] != self.config["batch_size"]*self.config["sequence_length"]:
                    continue

            images = images.to(device=DEVICE).reshape(self.config["batch_size"], self.config["sequence_length"], images.shape[-2], images.shape[-1])
            gt_seg = gt_seg.to(device=DEVICE).reshape(self.config["batch_size"], self.config["sequence_length"], gt_seg.shape[-2], gt_seg.shape[-1]).long()
            keypoints = keypoints.to(device=DEVICE).reshape(self.config["batch_size"]*self.config["sequence_length"], keypoints.shape[-2], keypoints.shape[-1])
            keypoints = keypoints.float().split(1, dim=0)
            keypoints = [keys[0][~torch.isnan(keys[0]).any(axis=1)][:, [1, 0]] for keys in keypoints]

            pred_seg = self.model(images).moveaxis(1, 2)
            softmax = pred_seg.softmax(dim=2)
            segmentation = softmax.argmax(dim=2)
            _, means, _ = self.localizer.estimate(softmax.flatten(0, 1), segmentation=torch.bitwise_or(segmentation == 2, segmentation == 3).flatten(0, 1))

            metric_dict = {}
            for metric in self.metrics:
                if metric.isKeypointMetric():
                    metric_dict[metric.name] = metric.compute(means, keypoints)
                else:
                    metric_dict[metric.name] = metric.compute(segmentation, gt_seg)
            
            loop.set_postfix(metric_dict)

    def get_final_metrics(self):
        return [metric.get_final_score() for metric in self.metrics]



# Input list of every model path that should be checked inside a class
# For example [[UNET_A, UNET_B, UNET_C], [OURS_A, OURS_B, OURS_C], [..]]
# Outputs [[[Precision, F1-Score, NME, IOU, DICE, InferenceSpeed], [..]] for every supplied model]
def evaluate_everything(checkpoints: List[List[str]], dataset_path: str, group_names: List[str]) -> List[List[List[float]]]:
    group_evals = []
    for MODEL_GROUP in checkpoints:
        per_model_evals = []
        for CHECKPOINT_PATH in MODEL_GROUP:
            if CHECKPOINT_PATH == "" or not os.path.isdir(CHECKPOINT_PATH):
                print("\033[93m" + "Please provide a viable checkpoint path")

            config = utils.load_config(os.path.join(CHECKPOINT_PATH, "config.yml"))
            config["dataset_path"] = dataset_path
            val_transforms = A.load(os.path.join(CHECKPOINT_PATH, "val_transform.yaml"), data_format='yaml')
            neuralNet = __import__(config["model"])
            model = neuralNet.Model(config, state_dict=torch.load(os.path.join(CHECKPOINT_PATH, "model.pth.tar"))).to(DEVICE)
            dataset = __import__('dataset').__dict__[config['dataset_name']]

    
            try:
                old_sequence_length = config["sequence_length"]
                config["sequence_length"] = 1
            except:
                old_sequence_length = 1
                config["sequence_length"] = 1


            val_ds = dataset(config, is_train=False, transform=val_transforms)
            val_loader = DataLoader(val_ds, 
                                    batch_size=config["batch_size"]*old_sequence_length,
                                    num_workers=2, 
                                    pin_memory=True, 
                                    shuffle=False)
            
            localizer = LSQLocalization(local_maxima_window = 11, 
                                        gauss_window = config["gauss_window"], 
                                        heatmapaxis = config["heatmapaxis"], 
                                        threshold = 0.5,
                                        device=DEVICE)
            
            config["sequence_length"] = old_sequence_length
            evaluator = None
            metrics = [AveragePrecisionMetric(), PrecisionMetric(), F1ScoreMetric(), NMEMetric(), ChamferMetric(), DiceMetric(), JaccardIndexMetric()]

            if config["model"] == "TwoDtoThreeDNet":
                evaluator = Evaluator2D3D(model, val_loader, localizer, config, metrics)
            else:
                evaluator = BaseEvaluator(model, val_loader, localizer, config, metrics)


            with torch.no_grad():
                print("#"*20)
                print("#"*3 + " " + CHECKPOINT_PATH + " " + "#"*3)
                print("#"*20)
                evaluator.evaluate()

                per_model_evals.append(evaluator.get_final_metrics())
        group_evals.append(per_model_evals)

    for group_name, group_scores in zip(group_names, group_evals):
        print("############" + group_name + "############")
        print("AveragePrecision, Precision, F1-Score, NME, Chamfer, DICE, IoU")
        print(torch.tensor(group_scores).mean(dim=0))
        print(torch.tensor(group_scores).std(dim=0))
        print()
        print()


def main():
    UNET_FULL = ["checkpoints/UNETFULL_CFCM_2558", 
                    "checkpoints/UNETFULL_DDFH_1010", 
                    #"checkpoints/UNETFULL_LS_RH_2342", 
                    "checkpoints/UNETFULL_MKMS_9400", 
                    "checkpoints/UNETFULL_SSTM_5445"]
    
    # UNET = ["checkpoints/UNET_CF_CM_3052", 
    #             "checkpoints/UNET_DD_FH_4761", 
    #             "checkpoints/UNET_LS_RH_2302", 
    #             "checkpoints/UNET_MK_MS_3426",
    #             "checkpoints/UNET_SS_TM_7862"]
    
    OURS = ["checkpoints/2D3D_CFCM_01_9160", 
                "checkpoints/2D3D_DDFH_01_3749", 
                "checkpoints/2D3D_LSRH_01_6746", 
                "checkpoints/2D3D_MKMS_01_3279", 
                "checkpoints/2D3D_SSTM_01_650"]
    
    #OURS_SGD = ["checkpoints/OURS_CFCM_SGD_9607", 
    #            "checkpoints/OURS_DDFH_SGD_1686", 
    #            "checkpoints/OURS_LSRH_SGD_6499", 
    #            "checkpoints/OURS_MKMS_SGD_1310", 
    #            "checkpoints/OURS_SSTM_SGD_7481"]
    
    MODEL_GROUPS = [UNET_FULL, OURS]#[UNET_FULL, UNET, OURS, OURS_SGD]
    MODEL_GROUP_NAMES = ["UNET_FULL", "OURS"]#["UNET_FULL", "UNET", "OURS", "OURS_SGD"]

    evaluate_everything(MODEL_GROUPS, '../HLEDataset/dataset/', MODEL_GROUP_NAMES)

    exit()
 
    parser = argparse.ArgumentParser(
                    prog = 'Inference for Deep Neural Networks',
                    description = 'Loads  as input, and visualize it based on the keys given in the config file.',
                    epilog = 'For question, generate an issue at: https://github.com/Henningson/SSSLsquared or write an E-Mail to: jann-ole.henningson@fau.de')
    parser.add_argument("-c", "--checkpoint", type=str, default="checkpoints/UNET_DD_FH_4761/")
    
    
    #OURS_DD_FH_1615/")
    
    #default="checkpoints/UNet_LSTM_Good/") Best LSTM Network, based on eval
    #default="checkpoints/2023-02-17-12:36:27_7564/") COMPARISON BASIC UNET + pretrain
    #default="checkpoints/2023-01-31-17:14:43_9554/") THE GOOD ONE
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
    with torch.no_grad():
        evaluate(val_loader, model, loss, localizer=localizer)

if __name__ == "__main__":
    main()