from typing import List, Union, Tuple, Optional
import torch
import torch.nn.functional as F

def topk_accuracy(
    output: torch.Tensor, 
    target: torch.Tensor, 
    topk: List[int]=(1,)
) -> List[torch.Tensor]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def precision(tp:torch.Tensor, fp:torch.Tensor, fn:Optional[torch.Tensor]=None) -> torch.Tensor:
    return tp / torch.clamp_min(tp + fp, 1)

def recall(tp:torch.Tensor, fp:torch.Tensor, fn:Optional[torch.Tensor]=None) -> torch.Tensor:
    if fn is None:
        return tp / torch.clamp_min(tp + fp, 1)
    else:
        return tp / torch.clamp_min(tp + fn, 1)

def dice_score(tp:torch.Tensor, fp:torch.Tensor, fn:torch.Tensor) -> torch.Tensor:
    return 2.0 * tp / torch.clamp_min(2.0 * tp + fp + fn, 1)

def f1_score(tp:torch.Tensor, fp:torch.Tensor, fn:torch.Tensor) -> torch.Tensor:
    p = precision(tp, fp, fn)
    r = recall(tp, fp, fn)
    return 2.0 * p * r / torch.clamp_min(p + r, 1e-10)

def average_precision(tp:torch.Tensor, fp:torch.Tensor, fn:torch.Tensor, dim:int=0) -> torch.Tensor:
    if tp.ndim == 2:
        raise ValueError("average_precision expects two dimensional inputs")

    r = recall(tp, fp, fn)
    p = recall(tp, fp, fn)
    return torch.sum((r[1:] - r[:-1])*p[1:])


def keypoint_statistics(
    prediction: List[torch.Tensor],
    target: List[torch.Tensor],
    threshold: Union[List[float], float],
    prediction_format: str = 'yx',
    target_format: str = 'xy'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    bs = len(prediction)

    if bs != len(target):
        raise RuntimeError(f"Batchsize of prediction and target must match. {bs} vs {len(target)}")

    sizes = [[len(kps_true), len(kps_pred)] for kps_true, kps_pred in zip(target, prediction)]
    

    # Targets to same length
    max_pred_keypoints = max(len(kps) for kps in prediction)
    kps_pred = torch.stack([
        F.pad(kps, (0, 0, 0, max_pred_keypoints - len(kps)), value=float('inf'))
        for kps in prediction
    ], axis=0)
    if prediction_format == 'yx':
        kps_pred = torch.flip(kps_pred, dims=[2])

    max_true_keypoints = max(len(kps) for kps in target)
    kps_true = torch.stack([
        F.pad(kps, (0, 0, 0, max_true_keypoints - len(kps)), value=float('inf'))
        for kps in target
    ])
    if target_format == 'yx':
        kps_true = torch.flip(kps_true, dims=[2])


    # Compute pairwise distances
    distance_matrices = torch.cdist(kps_true, kps_pred)

    true_to_pred_assignments = torch.full((bs, max_true_keypoints), -2, dtype=torch.int64)
    true_assigned_distances  = torch.full((bs, max_true_keypoints), float('inf'), dtype=torch.float32)
    pred_to_true_assignments = torch.full((bs, max_pred_keypoints), -2, dtype=torch.int64)
    assigned_distances       = torch.full((bs, max_pred_keypoints), float('inf'), dtype=torch.float32)

    for bid, (D, (num_true_keypoints, num_pred_keypoints)) in enumerate(zip(distance_matrices, sizes)):
        D = D[:num_true_keypoints, :num_pred_keypoints]
        
        true_to_pred_assignments[bid, :num_true_keypoints] = -1
        pred_to_true_assignments[bid, :num_pred_keypoints] = -1

        for _ in range(num_pred_keypoints):
            closest_idx = torch.argmin(D)
            true_idx, pred_idx = divmod(closest_idx.item(), num_pred_keypoints)

            match_not_yet_assigned = true_to_pred_assignments[bid, true_idx] == -1
            distance = D[true_idx, pred_idx]

            if match_not_yet_assigned:
                true_to_pred_assignments[bid, true_idx] = pred_idx
                pred_to_true_assignments[bid, pred_idx] = true_idx
                assigned_distances[bid, pred_idx] = distance
                true_assigned_distances[bid, true_idx] = distance
            D[:, pred_idx] = float('inf')

    threshold = torch.atleast_1d(torch.as_tensor(threshold)).to(assigned_distances.device)

    unassigned_gt  = torch.count_nonzero(true_to_pred_assignments == -1)
    true_distances = true_assigned_distances[true_to_pred_assignments >= 0]
    match_is_fp = true_distances[None, :] >= threshold[:, None]
    FN = torch.count_nonzero(match_is_fp, dim=1) + unassigned_gt 


    unmatched = torch.count_nonzero(pred_to_true_assignments == -1)
    assigned_distances = assigned_distances[pred_to_true_assignments >= 0]

    match_is_tp = assigned_distances[None, :] < threshold[:, None]

    TP = torch.count_nonzero(match_is_tp, dim=1)
    FP = len(assigned_distances) - TP + unmatched

    return TP, FP, FN         
