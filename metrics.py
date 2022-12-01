import torch
import numpy as np

# Given a number of estimated 2D Points and gt points
# Find for every gt point, an estimated point that lies inside a given threshold
class nnPrecision:
    def __init__(self, threshold = 3.0):
        self.threshold = threshold
        self.true_positives = []
        self.true_negatives = []
        self.false_positives = []
        self.false_negatives = []

    def __call__(self, pred, gt) -> float:
        for batch in pred.shape[0]:
            batch_pred = pred[batch]
            batch_gt = gt[batch]
            
            batched_tp = 0
            batched_tn = 0
            batched_fp = 0
            batched_fn = 0
            for i in range(batch_pred.shape[0]):
                index, distance = self.findNearestNeighbour(batch_pred[i], batch_gt)

                if distance < self.threshold:
                    batched_tp += 1


            return precision

    def findNearestNeighbour(self, point, target_points) -> int:
        if len(point.shape) == 1:
            point = point.unsqueeze(0)
        dist = torch.linalg.norm(point - target_points, dim=1)
        return torch.argmin(dist), dist.min()

if __name__ == "__main__":
    prec = nnPrecision()
    
    estimated = np.random.randn(2, 15)
    gt = np.random.randn(2, 13)