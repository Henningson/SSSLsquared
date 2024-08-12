import torch
import math
from torchmetrics.functional import dice, jaccard_index
import Metrics.KeypointMetrics as KeypointMetrics
import math
from chamferdist import ChamferDistance
import sys

sys.path.append("models/")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseMetric:
    def __init__(self, is_keypoint_metric=False):
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
        super().__init__(is_keypoint_metric=False)
        self.num_classes = num_classes
        self.name = "DICE"

    def compute(self, prediction, target) -> float:
        score = dice(prediction, target, num_classes=self.num_classes, ignore_index=0)
        self.running_average += score
        self.num_datapoints += 1
        return score


class JaccardIndexMetric(BaseMetric):
    def __init__(self, num_classes=4):
        super().__init__(is_keypoint_metric=False)
        self.num_classes = num_classes
        self.name = "IoU"

    def compute(self, prediction, target) -> float:
        score = jaccard_index(
            prediction, target, num_classes=self.num_classes, ignore_index=0
        )
        self.running_average += score
        self.num_datapoints += 1
        return score


class F1ScoreMetric(BaseMetric):
    def __init__(
        self, outlier_threshold=2.0, prediction_format="yx", target_format="yx"
    ):
        super().__init__(is_keypoint_metric=True)
        self.outlier_threshold = outlier_threshold
        self.prediction_format = prediction_format
        self.target_format = target_format
        self.TOTAL_TP = 0
        self.TOTAL_FP = 0
        self.TOTAL_TN = 0
        self.TOTAL_FN = 0
        self.name = "F1-Score"

    def compute(self, prediction, target) -> float:
        TP, FP, FN, _ = KeypointMetrics.keypoint_statistics(
            prediction,
            target,
            self.outlier_threshold,
            prediction_format=self.prediction_format,
            target_format=self.target_format,
        )
        self.TOTAL_TP += TP
        self.TOTAL_FP += FP
        self.TOTAL_FN += FN
        self.num_datapoints += 1
        return KeypointMetrics.f1_score(TP, FP, FN)

    def get_final_score(self):
        return KeypointMetrics.f1_score(self.TOTAL_TP, self.TOTAL_FP, self.TOTAL_FN)


class AveragePrecisionMetric(BaseMetric):
    def __init__(
        self, outlier_threshold=2.0, prediction_format="yx", target_format="yx"
    ):
        super().__init__(is_keypoint_metric=True)
        self.outlier_threshold = outlier_threshold
        self.prediction_format = prediction_format
        self.target_format = target_format
        self.TOTAL_TP = 0
        self.TOTAL_FP = 0
        self.TOTAL_TN = 0
        self.TOTAL_FN = 0
        self.name = "Precision"

    def compute(self, prediction, target) -> float:
        TP, FP, FN, _ = KeypointMetrics.keypoint_statistics(
            prediction,
            target,
            self.outlier_threshold,
            prediction_format=self.prediction_format,
            target_format=self.target_format,
        )
        self.TOTAL_TP += TP
        self.TOTAL_FP += FP
        self.TOTAL_FN += FN
        self.num_datapoints += 1
        return KeypointMetrics.average_precision(TP, FP, FN)

    def get_final_score(self):
        return KeypointMetrics.average_precision(
            self.TOTAL_TP, self.TOTAL_FP, self.TOTAL_FN
        )


class PrecisionMetric(BaseMetric):
    def __init__(
        self, outlier_threshold=2.0, prediction_format="yx", target_format="yx"
    ):
        super().__init__(is_keypoint_metric=True)
        self.outlier_threshold = outlier_threshold
        self.prediction_format = prediction_format
        self.target_format = target_format
        self.TOTAL_TP = 0
        self.TOTAL_FP = 0
        self.TOTAL_TN = 0
        self.TOTAL_FN = 0
        self.name = "Precision"

    def compute(self, prediction, target) -> float:
        TP, FP, FN, _ = KeypointMetrics.keypoint_statistics(
            prediction,
            target,
            self.outlier_threshold,
            prediction_format=self.prediction_format,
            target_format=self.target_format,
        )
        self.TOTAL_TP += TP
        self.TOTAL_FP += FP
        self.TOTAL_FN += FN
        self.num_datapoints += 1
        return KeypointMetrics.precision(TP, FP, FN)

    def get_final_score(self):
        return KeypointMetrics.precision(self.TOTAL_TP, self.TOTAL_FP, self.TOTAL_FN)


class NMEMetric(BaseMetric):
    def __init__(
        self, outlier_threshold=2.0, prediction_format="yx", target_format="yx"
    ):
        super().__init__(is_keypoint_metric=True)
        self.outlier_threshold = outlier_threshold
        self.prediction_format = prediction_format
        self.target_format = target_format
        self.total_l2_distances = []
        self.name = "NME"

    def compute(self, prediction, target) -> float:
        _, _, _, inlier_distances = KeypointMetrics.keypoint_statistics(
            prediction,
            target,
            self.outlier_threshold,
            prediction_format=self.prediction_format,
            target_format=self.target_format,
        )

        self.total_l2_distances = self.total_l2_distances + inlier_distances

        return sum(inlier_distances) / len(inlier_distances)

    def get_final_score(self):
        return sum(self.total_l2_distances) / len(self.total_l2_distances)


class ChamferMetric(BaseMetric):
    def __init__(self):
        super().__init__(is_keypoint_metric=True)
        self.name = "Chamfer"
        self.chamf = ChamferDistance()

    def compute(self, prediction, target) -> float:
        count = 0
        current_average = 0
        for i in range(len(prediction)):
            chamferdist = self.chamf(
                prediction[i].unsqueeze(0), target[i].unsqueeze(0), bidirectional=True
            ) / torch.sqrt(torch.tensor(512 * 512 + 256 * 256))
            if math.isnan(chamferdist):
                continue

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
            keypoints = [
                keys[0][~torch.isnan(keys[0]).any(axis=1)][:, [1, 0]]
                for keys in keypoints
            ]

            prediction = self.model(images).softmax(dim=1)
            segmentation = prediction.argmax(dim=1)
            _, means, _ = self.localizer.estimate(
                prediction,
                segmentation=torch.bitwise_or(segmentation == 2, segmentation == 3),
            )

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
            if (
                images.shape[0]
                != self.config["batch_size"] * self.config["sequence_length"]
            ):
                continue

            images = images.to(device=DEVICE).reshape(
                self.config["batch_size"],
                self.config["sequence_length"],
                images.shape[-2],
                images.shape[-1],
            )
            gt_seg = (
                gt_seg.to(device=DEVICE)
                .reshape(
                    self.config["batch_size"],
                    self.config["sequence_length"],
                    gt_seg.shape[-2],
                    gt_seg.shape[-1],
                )
                .long()
            )
            keypoints = keypoints.to(device=DEVICE).reshape(
                self.config["batch_size"] * self.config["sequence_length"],
                keypoints.shape[-2],
                keypoints.shape[-1],
            )
            keypoints = keypoints.float().split(1, dim=0)
            keypoints = [
                keys[0][~torch.isnan(keys[0]).any(axis=1)][:, [1, 0]]
                for keys in keypoints
            ]

            pred_seg = self.model(images).moveaxis(1, 2)
            softmax = pred_seg.softmax(dim=2)
            segmentation = softmax.argmax(dim=2)
            _, means, _ = self.localizer.estimate(
                softmax.flatten(0, 1),
                segmentation=torch.bitwise_or(
                    segmentation == 2, segmentation == 3
                ).flatten(0, 1),
            )

            metric_dict = {}
            for metric in self.metrics:
                if metric.isKeypointMetric():
                    metric_dict[metric.name] = metric.compute(means, keypoints)
                else:
                    metric_dict[metric.name] = metric.compute(segmentation, gt_seg)

            loop.set_postfix(metric_dict)

    def get_final_metrics(self):
        return [metric.get_final_score() for metric in self.metrics]
