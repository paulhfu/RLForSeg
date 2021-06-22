import sys
from rewards.reward_abc import RewardFunctionAbc
from skimage.measure import approximate_polygon, find_contours
from skimage.draw import polygon_perimeter, line
from skimage.transform import hough_circle, hough_circle_peaks
import torch
from skimage.draw import disk
import numpy as np
import matplotlib.pyplot as plt


class HoughCirclesReward(RewardFunctionAbc):

    def __init__(self, s_subgraph, *args, **kwargs):
        self.max_p = torch.nn.MaxPool2d(3, padding=1, stride=1)
        self.circle_thresh = 0.6
        self.range_rad = [10, 20]
        self.range_num = [20, 20]
        self.s_subgraph = s_subgraph

    def __call__(self, prediction_segmentation, superpixel_segmentation, dir_edges, subgraph_indices, *args, **kwargs):
        dev = prediction_segmentation.device
        edge_scores = []
        exp_factor = 3

        for single_pred, single_sp_seg, s_dir_edges in zip(prediction_segmentation, superpixel_segmentation, dir_edges):

            edge_score = torch.zeros(int((single_sp_seg.max()) + 1, ), device=dev)
            if single_pred.max() == 0:  # image is empty
                edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]
                edge_score = edge_score[edges].max(dim=0).values
                edge_scores.append(edge_score)
                continue
            # get one-hot representation
            one_hot = torch.zeros((int(single_pred.max()) + 1,) + single_pred.size(), device=dev, dtype=torch.long) \
                .scatter_(0, single_pred[None], 1)

            # need masses to determine what potential_objects can be considered background
            label_masses = one_hot.flatten(1).sum(-1)
            # everything else are potential potential_objects
            bg_obj_mask = label_masses > 1400
            potenial_obj_mask = label_masses <= 1400
            false_obj_mask = label_masses < 200
            bg_object_ids = torch.nonzero(bg_obj_mask).squeeze(1)  # object label IDs
            potential_object_ids = torch.nonzero(potenial_obj_mask).squeeze(1)  # object label IDs

            potential_objects = one_hot[potential_object_ids]  # get object masks
            bg_sp_ids = torch.unique((single_sp_seg[None] + 1) * one_hot[bg_object_ids])[1:] - 1
            object_sp_ids = [torch.unique((single_sp_seg[None] + 1) * obj)[1:] - 1 for obj in potential_objects]
            false_sp_ids = torch.unique((single_sp_seg[None] + 1) * one_hot[false_obj_mask])[1:] - 1

            # Detect two radii
            potential_fg = (potential_objects * torch.arange(len(potential_objects), device=dev)[:, None, None]).sum(0).float()
            edge_image = ((- self.max_p(-potential_fg.unsqueeze(0)).squeeze()) != potential_fg).float().cpu().numpy()
            hough_radii = np.arange(self.range_rad[0], self.range_rad[1])
            hough_res = hough_circle(edge_image, hough_radii)
            accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=self.range_num[1])
            mp_circles = torch.from_numpy(np.stack([cy, cx], axis=1))
            accepted_circles = accums > self.circle_thresh
            good_obj_cnt = 0

            if any(accepted_circles):
                mp_circles = mp_circles[accepted_circles]
                accums = accums[accepted_circles]
                circle_idxs = [disk(mp, rad, shape=single_sp_seg.shape) for mp, rad in zip(mp_circles, radii)]

                circle_sps = [torch.unique(single_sp_seg[circle_idx[0], circle_idx[1]]).long() for circle_idx in circle_idxs]
                obj_ids = [torch.unique(single_pred[circle_idx[0], circle_idx[1]]) for circle_idx in circle_idxs]

                for circle_sp, val, obj_id in zip(circle_sps, accums, obj_ids):
                    hough_score = (val - self.circle_thresh) / (1 - self.circle_thresh)
                    # hough_score = torch.sigmoid(torch.tensor([8 * (hough_score - 0.5)])).item()
                    num_obj_score = 1 / max(len(obj_id), 1)
                    if num_obj_score == 1 and obj_id[0] in potential_object_ids:
                        good_obj_cnt += 1
                    edge_score[circle_sp] = 0.7 * hough_score + 0.3 * num_obj_score

            score = 1.0 * (good_obj_cnt / 15) * int(good_obj_cnt > 5) + 0.0 * (1 / len(bg_object_ids))
            # score = 1 / len(bg_object_ids)
            score = np.exp((score * exp_factor)) / np.exp(np.array([exp_factor]))
            edge_score[bg_sp_ids] = score.item()
            edge_score[false_sp_ids] = 0.0
            if torch.isnan(edge_score).any() or torch.isinf(edge_score).any():
                print(Warning("NaN or inf in scores this should not happen"))
                sys.stdout.flush()
                assert False
            edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]
            edge_score = edge_score[edges].max(dim=0).values
            edge_scores.append(edge_score)

        edge_scores = torch.cat(edge_scores)
        sg_scores = []
        for i, sz in enumerate(self.s_subgraph):
            sg_scores.append(edge_scores[subgraph_indices[i].view(-1, sz)].mean(1))
        return sg_scores, edge_scores.mean()

