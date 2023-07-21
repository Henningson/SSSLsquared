import torch
import albumentations as A
import os
import Utils.utils as utils
from Metrics.Evaluator import * # Fix this
from Metrics.Timing import * # Fix this
from torch.utils.data import DataLoader
from models.LSQ import LSQLocalization
import torch
import albumentations as A
import os
import math
from torchmetrics.functional import dice, jaccard_index
import wandb
import Metrics.KeypointMetrics as KeypointMetrics
import math
from typing import List, Tuple
from chamferdist import ChamferDistance
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.LSQ import LSQLocalization
import sys
sys.path.append("models/")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# TODO: Reformulate to work similar to evaluate_everything(...).
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
            
            
            TP_temp, FP_temp, FN_temp, distances = KeypointMetrics.keypoint_statistics(pred_keypoints, gt_keypoints, 2.0, prediction_format="yx", target_format="yx")
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
            precision = KeypointMetrics.precision(TP, FP, FN)
            f1 = KeypointMetrics.f1_score(TP, FP, FN)
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





# Input list of every model path that should be checked inside a class
# For example [[UNET_A, UNET_B, UNET_C], [OURS_A, OURS_B, OURS_C], [..]]
# Outputs [[[Precision, F1-Score, NME, IOU, DICE, InferenceSpeed], [..]] for every supplied model]
def evaluate_everything(checkpoints: List[List[str]], dataset_path: str, group_names: List[str]) -> List[List[List[float]]]:
    group_evals = []
    group_times = []
    for MODEL_GROUP in checkpoints:
        per_model_evals = []
        per_model_times = []
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
            
            localizer = LSQLocalization(local_maxima_window = 7, 
                                        gauss_window = config["gauss_window"], 
                                        heatmapaxis = config["heatmapaxis"], 
                                        threshold = 0.7,
                                        device=DEVICE)
            
            config["sequence_length"] = old_sequence_length
            evaluator = None
            metrics = [PrecisionMetric(), F1ScoreMetric(), NMEMetric(), DiceMetric(), JaccardIndexMetric()]

            if config["model"] == "TwoDtoThreeDNet" or config["model"] == "TwoDtoThreeDNet2":
                timer_eval = InferenceTimer2D3D(model, val_loader, localizer, config)
                evaluator = Evaluator2D3D(model, val_loader, localizer, config, metrics)
            else:
                timer_eval = BaseInferenceTimer(model, val_loader, localizer, config)
                evaluator = BaseEvaluator(model, val_loader, localizer, config, metrics)


            with torch.no_grad():
                print("#"*20)
                print("#"*3 + " " + CHECKPOINT_PATH + " " + "#"*3)
                print("#"*20)
                timer_eval.evaluate()
                evaluator.evaluate()
                
                per_model_times.append(timer_eval.get_total_time())
                per_model_evals.append(evaluator.get_final_metrics())
        group_evals.append(per_model_evals)
        group_times.append(per_model_times)

    for group_name, group_scores, group_time in zip(group_names, group_evals, group_times):
        print("############" + group_name + "############")
        print("Precision, F1-Score, NME, DICE, IoU")
        print(torch.tensor(group_scores).mean(dim=0))
        print(torch.tensor(group_scores).std(dim=0))
        print("Inference-Time")
        print(torch.tensor(group_time).mean())
        print(torch.tensor(group_time).std())



def evaluate_single_network(checkpoint_path):
    # TODO: Implement me
    pass


def main():
    
    UNET_FULL = ["checkpoints/UNETFULL_CFCM_2558", 
                    "checkpoints/UNETFULL_DDFH_1010", 
                    "checkpoints/UNETFULL_LSRH_1355", 
                    "checkpoints/UNETFULL_MKMS_9400", 
                    "checkpoints/UNETFULL_SSTM_5445"]

    LSTM_UNET = ["checkpoints/LSTM_MK_MS_7289",
                "checkpoints/LSTM_CF_CM_5010", 
                "checkpoints/LSTM_DD_FH_7886", 
                "checkpoints/LSTM_LS_RH_8013", 
                "checkpoints/LSTM_SS_TM_6150"]

    ZweiDDreiD = ["checkpoints/2D3D_CFCM_01_9160", 
                "checkpoints/2D3D_DDFH_01_3749", 
                "checkpoints/2D3D_LSRH_01_6746", 
                "checkpoints/2D3D_MKMS_01_3279", 
                "checkpoints/2D3D_SSTM_01_650"]
    
    MODEL_GROUPS = [UNET_FULL, ZweiDDreiD, LSTM_UNET]
    MODEL_GROUP_NAMES = ["UNET_FULL", "ZweiDDreiD", "LSTM_UNET"]

    evaluate_everything(MODEL_GROUPS, '../HLEDataset/dataset/', MODEL_GROUP_NAMES)

if __name__ == "__main__":
    main()