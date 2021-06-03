import numpy as np
from skimage.metrics import contingency_table
from skimage.metrics import variation_of_information, adapted_rand_error
import threading

def precision(tp, fp, fn):
    return tp / (tp + fp) if tp > 0 else 0


def recall(tp, fp, fn):
    return tp / (tp + fn) if tp > 0 else 0


def accuracy(tp, fp, fn):
    return tp / (tp + fp + fn) if tp > 0 else 0


def f1(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0


def _relabel(input):
    _, unique_labels = np.unique(input, return_inverse=True)
    return unique_labels.reshape(input.shape)


def _iou_matrix(gt, seg):
    # relabel gt and seg for smaller memory footprint of contingency table
    gt = _relabel(gt)
    seg = _relabel(seg)

    # get number of overlapping pixels between GT and SEG
    n_inter = contingency_table(gt, seg).A

    # number of pixels for GT instances
    n_gt = n_inter.sum(axis=1, keepdims=True)
    # number of pixels for SEG instances
    n_seg = n_inter.sum(axis=0, keepdims=True)

    # number of pixels in the union between GT and SEG instances
    n_union = n_gt + n_seg - n_inter

    iou_matrix = n_inter / n_union
    # make sure that the values are within [0,1] range
    assert 0 <= np.min(iou_matrix) <= np.max(iou_matrix) <= 1

    return iou_matrix

class SegmentationMetrics:
    """
    Computes precision, recall, accuracy, f1 score for a given ground truth and predicted segmentation.
    Contingency table for a given ground truth and predicted segmentation is computed eagerly upon construction
    of the instance of `SegmentationMetrics`.
    Args:
        gt (ndarray): ground truth segmentation
        seg (ndarray): predicted segmentation
    """

    def __init__(self, gt, seg):
        self.iou_matrix = _iou_matrix(gt, seg)

    def metrics(self, iou_threshold):
        """
        Computes precision, recall, accuracy, f1 score at a given IoU threshold
        """
        # ignore background
        iou_matrix = self.iou_matrix[1:, 1:]
        detection_matrix = (iou_matrix > iou_threshold).astype(np.uint8)
        n_gt, n_seg = detection_matrix.shape

        # if the iou_matrix is empty or all values are 0
        trivial = min(n_gt, n_seg) == 0 or np.all(detection_matrix == 0)
        if trivial:
            tp = fp = fn = 0
        else:
            # count non-zero rows to get the number of TP
            tp = np.count_nonzero(detection_matrix.sum(axis=1))
            # count zero rows to get the number of FN
            fn = n_gt - tp
            # count zero columns to get the number of FP
            fp = n_seg - np.count_nonzero(detection_matrix.sum(axis=0))

        return {
            'precision': precision(tp, fp, fn),
            'recall': recall(tp, fp, fn),
            'accuracy': accuracy(tp, fp, fn),
            'f1': f1(tp, fp, fn)
        }

class AveragePrecision:
    """
    Average precision taken for the IoU range (0.5, 0.95) with a step of 0.05 as defined in:
    https://www.kaggle.com/stkbailey/step-by-step-explanation-of-scoring-metric
    """

    def __init__(self, iou=None):
        if iou is not None:
            self.iou_range = [iou]
        else:
            self.iou_range = np.linspace(0.50, 0.95, 10)

    def __call__(self, input_seg, gt_seg):
        if len(np.unique(gt_seg)) == 1:
            return 1.

        # compute contingency_table
        sm = SegmentationMetrics(gt_seg, input_seg)
        # compute accuracy for each threshold
        acc = [sm.metrics(iou)['accuracy'] for iou in self.iou_range]
        # return the average
        return np.mean(acc)

class ClusterMetrics:
    def __init__(self):
        self.splits_scores = []
        self.merges_scores = []
        self.are_score = []
        self.arp_score = []
        self.arr_score = []

    def reset(self):
        self.splits_scores = []
        self.merges_scores = []
        self.are_score = []
        self.arp_score = []
        self.arr_score = []

    def __call__(self, input_seg, gt_seg):
        splits, merges = variation_of_information(gt_seg, input_seg)
        self.splits_scores.append(splits)
        self.merges_scores.append(merges)
        are, arp, arr = adapted_rand_error(gt_seg, input_seg)
        self.are_score.append(are)
        self.arp_score.append(arp)
        self.arr_score.append(arr)

    def dump(self):
        return np.mean(self.splits_scores), np.mean(self.merges_scores), \
               np.mean(self.are_score), np.mean(self.arp_score), \
               np.mean(self.arr_score)

    def dump_std(self):
        return np.std(self.splits_scores), np.std(self.merges_scores), \
               np.std(self.are_score), np.std(self.arp_score), \
               np.std(self.arr_score)

import numpy as np
import nifty.ground_truth as ngt

# implementations based on:
# https://github.com/kreshuklab/sparse-object-embeddings/blob/master/pytorch3dunet/clustering/sbd.py


def dice_score(segmentation, groundtruth):
    """ Compute the dice score between binarized segmentation and ground-truth.
    Arguments:
        segmentation [np.ndarray] - candidate segmentation to evaluate
        groundtruth [np.ndarray] - groundtruth
    Returns:
        float - the dice score
    """
    seg = segmentation > 0
    gt = groundtruth > 0

    nom = 2 * np.sum(gt * seg)
    denom = np.sum(gt) + np.sum(seg)

    eps = 1e-7
    score = float(nom) / float(denom + eps)
    return score

def _best_dice_numpy(gt, seg):
    gt_labels = np.setdiff1d(np.unique(gt), [0])
    seg_labels = np.setdiff1d(np.unique(seg), [0])

    if len(seg_labels) == 0 or len(gt_labels) == 0:
        return 0.0

    best_dices = []
    for gt_idx in gt_labels:
        _gt_seg = (gt == gt_idx).astype('uint8')
        dices = []
        for pred_idx in seg_labels:
            _pred_seg = (seg == pred_idx).astype('uint8')

            dice = dice_score(_gt_seg, _pred_seg)
            dices.append(dice)
        best_dice = np.max(dices)
        best_dices.append(best_dice)

    return np.mean(best_dices)


def _best_dice_nifty(gt, seg):
    gt_labels, gt_counts = np.unique(gt, return_counts=True)
    seg_labels, seg_counts = np.unique(seg, return_counts=True)
    seg_counts = {seg_id: cnt for seg_id, cnt in zip(seg_labels, seg_counts)}

    if gt_labels[0] == 0:
        gt_labels, gt_counts = gt_labels[1:], gt_counts[1:]
    if len(gt_labels) == 0:
        return 0.0

    eps = 1e-7
    overlaps = ngt.overlap(gt, seg)
    dice_scores = []
    for gt_id, gt_count in zip(gt_labels, gt_counts):
        ovlp_ids, ovlp_counts = overlaps.overlapArrays(gt_id, sorted=True)
        if ovlp_ids[0] == 0:
            ovlp_ids, ovlp_counts = ovlp_ids[1:], ovlp_counts[1:]
        if len(ovlp_ids) == 0:
            dice_scores.append(0.)
            continue
        seg_id, count = ovlp_ids[0], ovlp_counts[0]
        score = float(2 * count) / float(gt_count + seg_counts[seg_id] + eps)
        dice_scores.append(score)

    return np.mean(dice_scores)


def symmetric_best_dice_score(segmentation, groundtruth, impl='nifty'):
    """ Compute the best symmetric dice score between the objects in the groundtruth and segmentation.
    This metric is used in the CVPPP instance segmentation challenge.
    Arguments:
        segmentation [np.ndarray] - candidate segmentation to evaluate
        groundtruth [np.ndarray] - groundtruth
        impl [str] - implementation used to compute the best dice score (default: 'nifty')
    Returns:
        float - the best symmetric dice score
    """
    assert impl in ('nifty', 'numpy')
    best_dice = _best_dice_nifty if impl == 'nifty' else _best_dice_numpy
    score1 = best_dice(segmentation, groundtruth)
    score2 = best_dice(groundtruth, segmentation)
    return min(score1, score2)

if __name__ == "__main__":
    metric = AveragePrecision()
    cluster_metrics = ClusterMetrics()
    y_true = np.zeros((100,100), np.uint16)
    y_true[10:20,10:20] = 1
    y_pred = np.roll(y_true, 10, axis = 0) * 2
    score = metric(y_pred, y_true)
    cluster_metrics(y_pred, y_true)
    cl_scores = cluster_metrics.dump()

    print(score)
    print(*cl_scores)