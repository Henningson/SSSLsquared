import torch
import sys
from tqdm import tqdm
sys.path.append("models/")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class InferenceTimer2D3D:
    def __init__(self, model, val_loader, localizer, config, warm_up = 3):
        self.model = model
        self.val_loader = val_loader
        self.config = config
        self.localizer = localizer
        self.warm_up = warm_up

        self.total_inference_time = 0.0
        self.total_point_prediction_time = 0.0
        self.num_images = 0

    def start_timer(self):
        torch.cuda.synchronize()
        self.timer_start = torch.cuda.Event(enable_timing=True)
        self.timer_end = torch.cuda.Event(enable_timing=True)
        self.timer_start.record()

    def stop_timer(self):
        self.timer_end.record()
        torch.cuda.synchronize()
        return self.timer_start.elapsed_time(self.timer_end)

    def evaluate(self):
        loop = tqdm(self.val_loader, desc="Evaluation")
        count = 0
        for images, gt_seg, keypoints in loop:
            if images.shape[0] != self.config["batch_size"]*self.config["sequence_length"]:
                    continue

            images = images.to(device=DEVICE).reshape(self.config["batch_size"], self.config["sequence_length"], images.shape[-2], images.shape[-1])
            gt_seg = gt_seg.to(device=DEVICE).reshape(self.config["batch_size"], self.config["sequence_length"], gt_seg.shape[-2], gt_seg.shape[-1]).long()
            keypoints = keypoints.to(device=DEVICE).reshape(self.config["batch_size"]*self.config["sequence_length"], keypoints.shape[-2], keypoints.shape[-1])
            keypoints = keypoints.float().split(1, dim=0)
            keypoints = [keys[0][~torch.isnan(keys[0]).any(axis=1)][:, [1, 0]] for keys in keypoints]

            self.start_timer()
            pred_seg = self.model(images).moveaxis(1, 2)
            softmax = pred_seg.softmax(dim=2)
            segmentation = softmax.argmax(dim=2)
            inference_time = self.stop_timer()

            self.start_timer()
            _, means, _ = self.localizer.estimate(softmax.flatten(0, 1), segmentation=torch.bitwise_or(segmentation == 2, segmentation == 3).flatten(0, 1))
            point_detection_time = self.stop_timer()

            if count > self.warm_up:
                self.num_images += pred_seg.flatten(0, 1).shape[0]
                self.total_inference_time += inference_time
                self.total_point_prediction_time += point_detection_time

            count += 1
            loop.set_postfix({"InferenceTime": inference_time, "LocalizeTime": point_detection_time})

    def get_total_time(self):
        return (self.total_inference_time + self.total_point_prediction_time) / self.num_images



class BaseInferenceTimer:
    def __init__(self, model, val_loader, localizer, config, warm_up = 3):
        self.model = model
        self.val_loader = val_loader
        self.config = config
        self.localizer = localizer
        self.warm_up = warm_up

        self.total_inference_time = 0.0
        self.total_point_prediction_time = 0.0
        self.num_images = 0

    def start_timer(self):
        torch.cuda.synchronize()
        self.timer_start = torch.cuda.Event(enable_timing=True)
        self.timer_end = torch.cuda.Event(enable_timing=True)
        self.timer_start.record()

    def stop_timer(self):
        self.timer_end.record()
        torch.cuda.synchronize()
        return self.timer_start.elapsed_time(self.timer_end)

    def evaluate(self):
        loop = tqdm(self.val_loader, desc="Evaluation")
        count = 0
        for images, gt_seg, keypoints in loop:
            images = images.to(device=DEVICE)
            gt_seg = gt_seg.to(device=DEVICE).long()
            keypoints = keypoints.to(device=DEVICE)
            
            keypoints = keypoints.float().split(1, dim=0)
            keypoints = [keys[0][~torch.isnan(keys[0]).any(axis=1)][:, [1, 0]] for keys in keypoints]

            self.start_timer()
            prediction = self.model(images).softmax(dim=1)
            segmentation = prediction.argmax(dim=1)
            inference_time = self.stop_timer()

            self.start_timer()
            _, means, _ = self.localizer.estimate(prediction, segmentation=torch.bitwise_or(segmentation == 2, segmentation == 3))
            point_detection_time = self.stop_timer()

            if count > self.warm_up:
                self.num_images += prediction.shape[0]
                self.total_inference_time += inference_time
                self.total_point_prediction_time += point_detection_time

            count += 1
            loop.set_postfix({"InferenceTime": inference_time, "LocalizeTime": point_detection_time})

    def get_total_time(self):
        return (self.total_inference_time + self.total_point_prediction_time) / self.num_images