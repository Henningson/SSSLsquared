from dataset import HLEPlusPlus
import metrics_dom
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import albumentations as A
import kornia
import math

def main():

    config = {
        'dataset_path': '../HLEDataset/dataset/',
        'val_keys': "CF,CM,DD,FH,LS,RH,SS,TM,MK,MS",
        'train_keys': '',
        'pad_keypoints': 200,
    }
    # load Data
    dataset = HLEPlusPlus(config, is_train=False, transform=A.load("val_transform.yaml", data_format='yaml'))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    precision_list = []
    nme_list = []
    f1_list = []
    TP = 0
    FP = 0
    FN = 0

    for image, seg, keypoints in tqdm(dataloader):
        seg = seg.long().squeeze()
        seg = torch.where(seg == 1, 0, seg)
        seg = torch.where(seg == 2, 1, seg)
        seg = torch.where(seg == 3, 1, seg)
        seg = seg.squeeze()

        image = image.squeeze()[0]
        image -= image.min()
        image /= image.max()
    
        keypoints = keypoints.float()
        gt_keypoints = keypoints.split(1, dim=0)
        gt_keypoints = [keys[0][~torch.isnan(keys[0]).any(axis=1)][:, [1, 0]] for keys in gt_keypoints][0]
    
        points = locate_points(image, seg)
        refined_points = refine_points(image, points)

        # use dominiks stuff here to evaluate
        TP_temp, FP_temp, FN_temp, distances = metrics_dom.keypoint_statistics(refined_points.unsqueeze(0), gt_keypoints.unsqueeze(0), 2.0, prediction_format="yx", target_format="yx")
        TP += TP_temp
        FP += FP_temp
        FN += FN_temp

        precision = metrics_dom.precision(TP, FP, FN)
        f1 = metrics_dom.f1_score(TP, FP, FN)

        precision_list.append(precision)
        f1_list.append(f1)
        for distance in distances:
            nme_list.append(distance)


    print("Precision: {0}".format(sum(precision_list) / len(precision_list)))
    print("F1: {0}".format(sum(f1_list) / len(f1_list)))
    print("NME: {0}".format(sum(nme_list) / len(nme_list)))

    





def locate_points(image, vocalfold_segmentation, kernelsize=5):
    kernel = torch.ones((kernelsize, kernelsize))
    kernel[math.floor(kernelsize//2), math.floor(kernelsize//2)] = 0.0
    
    dilated_image = kornia.morphology.dilation(image[None, None, :, :], kernel).squeeze()
    local_maxima = (image > dilated_image)*1
    local_maxima *= vocalfold_segmentation
    return local_maxima.nonzero()

def refine_points(image, points, window_size=5):
    temp_points = points.float()
    for i in range(points.shape[0]):
        current_point = points[i]
        
        # Span Window
        temp_point = image[current_point[0] - window_size//2:current_point[0] + window_size//2 + 1,
                            current_point[1] - window_size//2:current_point[1] + window_size//2 + 1]
        
        
        # Build coordinates for each pixel of window
        sub = torch.linspace(-window_size//2 + 1, window_size//2, window_size)
        x_sub, y_sub = torch.meshgrid(sub, sub, indexing="xy")
        x_offset = (x_sub*temp_point).sum() / temp_point.sum()
        y_offset = (y_sub*temp_point).sum() / temp_point.sum()

        current_point = current_point.float()
        current_point[0] += y_offset
        current_point[1] += x_offset

        temp_points[i] = current_point

    return temp_points


if __name__ == "__main__":
    main()