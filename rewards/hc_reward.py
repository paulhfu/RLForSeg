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

class HoneycombReward(RewardFunctionAbc):

    def __init__(self, shape_samples):
        #TODO get the descriptors for the shape samples
        dev = shape_samples.device

        self.samples = []
        self.false_obj_min = 15
        self.false_obj_max = 70**2
        self.valid_metric = AveragePrecision()

        shape_sample = shape_samples[0]
        shape_sample = shape_sample.cpu().numpy().astype(np.int32)
        b_box = scipy.ndimage.find_objects(shape_sample)[0]
        crop = shape_sample[b_box[0].start - 1:b_box[0].stop + 1,
                            b_box[1].start - 1:b_box[1].stop + 1]
        self.samples.append(crop)
        self.default_reward = 0.1

    def __call__(self, pred_segm, sp_segm, res, dir_edges,
                 edge_score, *args, **kwargs):
        dev = pred_segm.device
        return_scores = []
        inner_halo_mask = torch.zeros(sp_segm.shape[1:], device=dev)
        inner_halo_mask[0, :] = 1
        inner_halo_mask[:, 0] = 1
        inner_halo_mask[-1, :] = 1
        inner_halo_mask[:, -1] = 1
        inner_halo_mask = inner_halo_mask.unsqueeze(0)

        for single_pred, single_sp_seg, s_dir_edges in zip(pred_segm, sp_segm, dir_edges):
            scores = torch.ones(int((single_sp_seg.max()) + 1,), device=dev) * self.default_reward

            if single_pred.max() == 0:  # image is empty
                scores -= 0.5
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
            bg_mask = torch.zeros_like(label_masses).bool()
            bg_id = label_masses.argmax()
            bg_mask[bg_id] = True

            # get the objects that are touching the patch boarder as for them we cannot
            # compute a reliable sim score
            invalid_obj_mask = ((one_hot * inner_halo_mask).flatten(1).sum(-1) >= 2) & (bg_mask == False)
            # Apply size fitler and invalidate those which dont fall within a range
            false_obj_mask = (label_masses < self.false_obj_min) | (label_masses > self.false_obj_max)
            false_obj_mask[bg_id] = False
            # everything else are potential objects
            potenial_obj_mask = (false_obj_mask == False) & (invalid_obj_mask == False)
            potenial_obj_mask[bg_id] = False
            potential_object_ids = torch.nonzero(potenial_obj_mask).squeeze(1)  # object label IDs

            bg = one_hot[bg_id]  # get background masks
            objects = one_hot[potential_object_ids]  # get object masks
            false_obj_sp_ids = torch.unique((single_sp_seg[None] + 1) * one_hot[false_obj_mask])[1:] - 1
            # mask out the covered superpixels (need to add 1 because the single_sp_seg start from 0)
            bg_sp_ids = [torch.unique((single_sp_seg[None] + 1) * bg_obj)[1:] - 1 for bg_obj in bg]
            object_sp_ids = [torch.unique((single_sp_seg[None] + 1) * obj)[1:] - 1 for obj in objects]

            #get shape descriptors for objects and get a score by comparing to self.descriptors
            for obj, sp_ids in zip(objects, object_sp_ids):
                obj = obj.cpu().numpy()
                b_box = scipy.ndimage.find_objects(obj)[0]
                crop = obj[b_box[0].start - 1:b_box[0].stop + 1,
                            b_box[1].start - 1:b_box[1].stop + 1]
                #contour = find_contours(crop, level=0)[0]
                #mass_center = np.mean(contour, axis=0)

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
                pred_cont = find_contours(pred_padded, level=0)
                if (len(pred_cont) != 0):
                    contour = pred_cont[0]
                    pred_cm = np.mean(contour, axis=0, dtype=np.int32)
                    contour = find_contours(gt_padded, level=0)[0]
                    gt_cm = np.mean(contour, axis=0, dtype=np.int32)

                    roll_y, roll_x = int(gt_cm[0] - pred_cm[0]), int(gt_cm[1] - pred_cm[1])
                    pred_padded = np.roll(pred_padded, (roll_y, roll_x))

                    score = self.valid_metric(pred_padded, gt_padded)
                    scores[sp_ids] = score
                else:
                    scores[sp_ids] = 0

                #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
                #ax1.imshow(gt_padded, interpolation='none')
                #pred_padded[pred_cm[0], pred_cm[1]] = 2
                #ax2.imshow(pred_padded, interpolation='none')
                '''
                poly_chain = torch.from_numpy(approximate_polygon(contour, tolerance=0.0)).to(dev)
                print("Polychain appr", poly_chain.shape)

                if poly_chain.shape[0] <= 5:
                    scores[sp_ids] -= 0.1
                    continue
                polygon = Polygon2d(poly_chain)
                dist_scores = torch.tensor([des.distance(polygon, res) for des in self.gt_descriptors], device=dev)
                print("dist_score", dist_scores)
                #project distances for objects to similarities for superpixels
                scores[sp_ids] += 1 - dist_scores.min()
                # do exponential scaling
                # scores[sp_ids] += torch.exp((1 - dist_scores.min()) * 8) /
                #                             torch.exp(torch.tensor([8.0], device=dev))
                '''
                if torch.isnan(scores).any() or torch.isinf(scores).any():
                    a=1

            scores[false_obj_sp_ids] -= self.default_reward
            # get score for the background
            # would be nice if this was not necessary. So first see and check if it works without
            # if len(object_ids) <= 3:
            #     for bg_sp_id in bg_sp_ids:
            #         scores[bg_sp_id] -= .5
            #
            # if len(bg_id) >= 2:
            #     for bg_sp_id in bg_sp_ids:
            #         scores[bg_sp_id] -= .5
            # else:
            #     for bg_sp_id in bg_sp_ids:
            #         scores[bg_sp_id] += .2

            if edge_score:
                edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]
                edge_scores = scores[edges].max(dim=0).values
                return_scores.append(edge_scores)
            else:
                return_scores.append(scores)
            #return scores for each superpixel

        return torch.cat(return_scores)

if __name__ == "__main__":
    dev = "cuda:0"
    # get a few images and extract some gt objects used ase shape descriptors that we want to compare against
    fnames_pix = sorted(glob('/g/kreshuk/kaziakhm/toy_data_honeycombs/train/pix_data/*.h5'))
    fnames_graph = sorted(glob('/g/kreshuk/kaziakhm/toy_data_honeycombs/train/graph_data/*.h5'))
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
    f = HoneycombReward(sample_shapes[5:6,...])
