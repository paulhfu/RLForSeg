from rewards.reward_abc import RewardFunctionAbc
from skimage.measure import approximate_polygon,  find_contours
from skimage.draw import polygon_perimeter
from utils.polygon_2d import Polygon2d
from cv2 import fitEllipse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import skimage
import scipy.ndimage
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from utils.metrics import AveragePrecision

import h5py
import numpy as np
import matplotlib.pyplot as plt
from utils.general import multicut_from_probas
from glob import glob
import os
import torch

class HoneycombRewardv2(RewardFunctionAbc):

    def __init__(self, shape_samples):
        dev = shape_samples.device

        self.samples = []
        self.valid_metric = AveragePrecision()

        shape_sample = shape_samples[0]
        shape_sample = shape_sample.cpu().numpy().astype(np.int32)
        b_box = scipy.ndimage.find_objects(shape_sample)[0]
        crop = shape_sample[b_box[0].start - 1:b_box[0].stop + 1,
                            b_box[1].start - 1:b_box[1].stop + 1]

        gt_mass = len(crop[crop > 0])
        self.size_cutoff = 0.8
        self.size_mean = torch.tensor(gt_mass)
        self.size_sigma = torch.tensor(0.025 * gt_mass)

        self.samples.append(crop)

    def gaussian(self, x, mu, sig):
        return torch.exp(-torch.pow(x - mu, 2.) / (2 * torch.pow(sig, 2.))).type(torch.FloatTensor)

    def __call__(self, pred_segm, sp_segm, res, dir_edges,
                 edge_score, *args, **kwargs):
        dev = pred_segm.device
        return_scores = []

        for single_pred, single_sp_seg, s_dir_edges in zip(pred_segm, sp_segm, dir_edges):
            scores = torch.zeros(int((single_sp_seg.max()) + 1,), device=dev)

            if single_pred.max() == 0:  # image is empty
                if edge_score:
                    edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]
                    edge_scores = scores[edges].max(dim=0).values
                    return_scores.append(edge_scores)
                else:
                    return_scores.append(scores)
                continue

            # get one-hot representation
            one_hot = torch.zeros((int(single_pred.max()) + 1, ) + single_pred.size(),
                                  device=dev, dtype=torch.long).scatter_(0, single_pred[None], 1)
            # need masses to determine what objects can be considered background
            label_masses = one_hot.flatten(1).sum(-1)
            size_mask = self.gaussian(label_masses, self.size_mean, self.size_sigma)
            # get the objects that are touching the patch boarder as for them we cannot
            # compute a reliable sim score
            # Apply size fitler and invalidate those which dont fall within a range
            false_obj_mask = (size_mask < self.size_cutoff)
            # everything else are potential objects
            potenial_obj_mask = (false_obj_mask == False)
            potential_object_ids = torch.nonzero(potenial_obj_mask).squeeze(1)  # object label IDs

            objects = one_hot[potential_object_ids]  # get object masks
            false_obj_index = torch.nonzero(false_obj_mask == True).squeeze(1)
            false_obj_sp_ids = torch.unique((single_sp_seg[None] + 1) * one_hot[false_obj_mask])[1:] - 1
            # mask out the covered superpixels (need to add 1 because the single_sp_seg start from 0)
            object_sp_ids = [torch.unique((single_sp_seg[None] + 1) * obj)[1:] - 1 for obj in objects]

            #get shape descriptors for objects and get a score by comparing to self.descriptors
            for obj, sp_ids, index in zip(objects, object_sp_ids, potential_object_ids):
                obj = obj.cpu().numpy()
                b_box = scipy.ndimage.find_objects(obj)[0]
                crop = obj[max(b_box[0].start - 1, 0):min(b_box[0].stop + 1, obj.shape[0]),
                           max(b_box[1].start - 1, 0):min(b_box[1].stop + 1, obj.shape[1])]

                # Pad the crop and gt_sample so that there is a room for rolling
                max_w = max(crop.shape[1], self.samples[0].shape[1]) * 3
                max_h = max(crop.shape[0], self.samples[0].shape[0]) * 3

                pad_w1, pad_h1 = (max_w - crop.shape[1])//2, (max_h - crop.shape[0])//2
                pad_w2, pad_h2 = (max_w - self.samples[0].shape[1])//2,\
                                 (max_h - self.samples[0].shape[0])//2
                diff_h = max_h - 2*pad_h1 - crop.shape[0]
                diff_w = max_w - 2*pad_w1 - crop.shape[1]
                pred_padded = np.pad(crop, [[pad_h1, pad_h1+diff_h], [pad_w1, pad_w1+diff_w]],
                                     mode='constant', constant_values=0)
                diff_h = max_h - 2*pad_h2 - self.samples[0].shape[0]
                diff_w = max_w - 2*pad_w2 - self.samples[0].shape[1]
                gt_padded = np.pad(self.samples[0], [[pad_h2, pad_h2+diff_h], [pad_w2, pad_w2+diff_w]],
                                   mode='constant', constant_values=0)

                # Roll some of the images so the centers of mass are aligned
                contour = find_contours(pred_padded, level=0)[0]
                pred_cm = np.mean(contour, axis=0, dtype=np.int32)
                contour = find_contours(gt_padded, level=0)[0]
                gt_cm = np.mean(contour, axis=0, dtype=np.int32)

                roll_y, roll_x = int(gt_cm[0] - pred_cm[0]), int(gt_cm[1] - pred_cm[1])
                pred_padded = np.roll(pred_padded, (roll_y, roll_x))

                score = self.valid_metric(pred_padded, gt_padded)
                scores[sp_ids] = (score + size_mask[index].cpu().numpy())/2

                #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
                #ax1.imshow(gt_padded, interpolation='none')
                #pred_padded[pred_cm[0], pred_cm[1]] = 2
                #ax2.imshow(pred_padded, interpolation='none')

                if torch.isnan(scores).any() or torch.isinf(scores).any():
                    a=1

            for false_sp_ids, index in zip(false_obj_sp_ids, false_obj_index):
                scores[false_sp_ids] = size_mask[index]

            if edge_score:
                edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]
                edge_scores = scores[edges].max(dim=0).values
                return_scores.append(edge_scores)
            else:
                return_scores.append(scores)

        return torch.cat(return_scores)

if __name__ == "__main__":
    dev = "cuda:0"
    # get a few images and extract some gt objects used ase shape descriptors that we want to compare against
    fnames_pix = sorted(glob('/g/kreshuk/kaziakhm/toy_data_honeycombs_v2_n3/train/pix_data/*.h5'))
    fnames_graph = sorted(glob('/g/kreshuk/kaziakhm/toy_data_honeycombs_v2_n3/train/graph_data/*.h5'))
    TEST_SAMPLES = 42

    gt = torch.from_numpy(h5py.File(fnames_pix[TEST_SAMPLES], 'r')['gt'][:]).to(dev)

    # set gt to integer labels
    _gt = torch.zeros_like(gt).long()
    for _lbl, lbl in enumerate(torch.unique(gt)):
        _gt += (gt == lbl).long() * _lbl
    gt = _gt
    # 0 should be background
    sample_shapes = torch.zeros((int(gt.max()) + 1,) + gt.size(), device=dev).scatter_(0, gt[None], 1)[1:]

    print("Sample shapes", sample_shapes.shape)
    f = HoneycombReward(sample_shapes[6:7,...])
