import torch
import numpy as np

# Given a number of estimated 2D Points and gt points
# Find for every gt point, an estimated point that lies inside a given threshold
class nnPrecision:
    def __init__(self, threshold = 3.0):
        self.threshold = threshold
        self.precisions = []

    def __call__(self, pred, gt) -> float:
        for batch in range(len(pred)):
            batch_pred = pred[batch]
            batch_gt = gt[batch]
            
            batched_tp = 0
            batched_fp = 0
            for i in range(batch_gt.shape[0]):
                _, distance = self.findNearestNeighbour(batch_gt[i], batch_pred)
                if distance < self.threshold:
                    batched_tp += 1
                else:
                    batched_fp += 1

        precision = batched_tp / (batched_tp + batched_fp)
        self.precisions.append(precision)
        return precision

    def findNearestNeighbour(self, point, target_points) -> int:
        if len(point.shape) == 1:
            point = point.unsqueeze(0)
        dist = torch.linalg.norm(point - target_points, dim=1)
        return torch.argmin(dist), dist.min()

    def compute(self):
        return sum(self.precisions) / len(self.precisions)


class nnMSE:
    def __init__(self, threshold = 3.0):
        self.threshold = threshold
        self.distances = []

    def __call__(self, pred, gt) -> float:
        batched_distance = 0
        neighbours = 0

        for batch in range(len(pred)):
            batch_pred = pred[batch]
            batch_gt = gt[batch]
            for i in range(batch_gt.shape[0]):
                _, distance = self.findNearestNeighbour(batch_gt[i], batch_pred)
                
                if distance < self.threshold:
                    batched_distance += distance
                    neighbours += 1

        l2 = batched_distance / neighbours
        self.distances.append(l2)
        return l2

    def findNearestNeighbour(self, point, target_points) -> int:
        if len(point.shape) == 1:
            point = point.unsqueeze(0)
        dist = torch.linalg.norm(point - target_points, dim=1)
        return torch.argmin(dist), dist.min()

    def compute(self):
        return sum(self.distances) / len(self.distances)

if __name__ == "__main__":
    Precision = nnMSE(threshold=0.5)
    
    estimated = [torch.randn(15, 2)]
    gt = [torch.randn(13, 2)]
    print(gt)
    print(Precision(estimated, gt))
    print(Precision.compute())
