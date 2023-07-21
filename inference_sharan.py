import torch
import albumentations as A
import os
import Utils.utils as utils
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from copy import deepcopy
from scipy import ndimage
import skimage.draw
import skimage.measure
import skimage.morphology
import Metrics.KeypointMetrics as KeypointMetrics

import sys
sys.path.append("models/")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def class_to_color(prediction, class_colors):
    prediction = np.expand_dims(prediction, 1)
    output = np.zeros((prediction.shape[0], 3, prediction.shape[-2], prediction.shape[-1]), dtype=np.float32)
    for class_idx, color in enumerate(class_colors):
        mask = class_idx == prediction
        curr_color = color.reshape(1, 3, 1, 1)
        segment = mask*curr_color # should have shape 1, 3, 100, 100
        output += segment

    return output


def centres_of_mass(mask, threshold):
    mask_bin = deepcopy(mask)
    mask_bin[mask >= threshold] = 1
    mask_bin[mask < threshold] = 0

    label_regions, num_regions = skimage.measure.label(mask_bin, background=0, return_num=True)
    indexlist = [item for item in range(1, num_regions + 1)]
    return ndimage.measurements.center_of_mass(mask_bin, label_regions, indexlist)


def evaluate(checkpoint_path, dataset_path = "../HLEDataset/dataset/"):
    if checkpoint_path == "" or not os.path.isdir(checkpoint_path):
        print("\033[93m" + "Please provide a viable checkpoint path")

    config = utils.load_config(os.path.join(checkpoint_path, "config.yml"))
    config['dataset_path'] = dataset_path

    val_transforms = A.Compose([
                        A.ToFloat(),
                        A.Normalize(),
                        A.pytorch.transforms.ToTensorV2()])

    neuralNet = __import__(config["model"])
    model = neuralNet.Model(config).to(DEVICE)
    state = torch.load(os.path.join(checkpoint_path, "model.pth.tar"))
    del state["optimizer"]

    model.load_state_dict(state)
    dataset = __import__('dataset').__dict__[config['dataset_name']]
    val_ds = dataset(config, is_train=False, transform=val_transforms)

    val_loader = DataLoader(val_ds, 
                            batch_size=config['batch_size'],
                            num_workers=2, 
                            pin_memory=True, 
                            shuffle=False)

    model.eval()
    TP = 0
    FP = 0
    FN = 0
    inference_time = 0
    l2_distances = []
    iters = 0
    count = 0
    for images, gt_seg, gt_heat, gt_keypoints in tqdm(val_loader, desc="Generating Video Frames"):
        images = images.to(device=DEVICE)
        gt_seg = gt_seg.to(device=DEVICE)
        
        gt_keypoints = gt_keypoints.float()
        gt_keypoints = gt_keypoints.split(1, dim=0)
        gt_keypoints = [keys[0][~torch.isnan(keys[0]).any(axis=1)][:, [1, 0]] for keys in gt_keypoints]

        torch.cuda.synchronize()
        starter_cnn, ender_cnn = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
        starter_cnn.record()
        binary, logits = model(images)
        ender_cnn.record()
        torch.cuda.synchronize()

        if iters > 10:
            inference_time += starter_cnn.elapsed_time(ender_cnn)
            count += logits.shape[0]

        points = [torch.tensor(centres_of_mass(logits[i, 0].detach().cpu().numpy(), 0.5), dtype=torch.float32) for i in range(logits.shape[0])]


        try:
            TP_temp, FP_temp, FN_temp, distances = KeypointMetrics.keypoint_statistics(points, gt_keypoints, 2.0, prediction_format="yx", target_format="yx")
            TP += TP_temp
            FP += FP_temp
            FN += FN_temp
            l2_distances = l2_distances + distances
        except:
            pass

        iters += 1

    ap = KeypointMetrics.average_precision(TP, FP, FN)
    f1 = KeypointMetrics.f1_score(TP, FP, FN)
    dice = KeypointMetrics.dice_score(TP, FP, FN)
    recoll = KeypointMetrics.recall(TP, FP, FN)
    prec = KeypointMetrics.precision(TP, FP, FN)
    inf_time = inference_time/count

    print("Precision: {0}, F1: {1}, DICE: {2}, Recall: {3}, Precision: {4}, Inf. Time: {5}, FPS: {6}".format(prec, f1, dice, recoll, prec, inf_time, 1000/inf_time))
    return prec, f1

if __name__ == "__main__":
    prec_0, f1_0 = evaluate("checkpoints/SHARAN_CFCM/")
    prec_1, f1_1 = evaluate("checkpoints/SHARAN_DDFH/")
    prec_2, f1_2 = evaluate("checkpoints/SHARAN_LSRH/")
    prec_3, f1_3 = evaluate("checkpoints/SHARAN_MKMS/")
    prec_4, f1_4 = evaluate("checkpoints/SHARAN_SSTM/")

    prec = torch.tensor([prec_0, prec_1, prec_2, prec_3, prec_4])
    print("Precsision: {0:04f}".format(prec.mean()))
    print("Precsision STD: {0:04f}".format(prec.std()))

    f1 = torch.tensor([f1_0, f1_1, f1_2, f1_3, f1_4])
    print("Precsision: {0:04f}".format(f1.mean()))
    print("Precsision STD: {0:04f}".format(f1.std()))