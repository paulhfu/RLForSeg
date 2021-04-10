import matplotlib
import sys
# matplotlib.use('TkAgg')
from rewards.reward_abc import RewardFunctionAbc
from skimage.measure import approximate_polygon, find_contours
from skimage.morphology import dilation
from skimage.draw import polygon_perimeter, line
from utils.polygon_2d import Polygon2d
from utils.general import random_label_cmap, get_colored_edges_in_sseg, sync_segmentations, get_contour_from_2d_binary
from utils.graphs import get_angles_smass_in_rag
from skimage.transform import hough_circle, hough_circle_peaks
import torch
from skimage.draw import disk
import math
import copy
from elf.segmentation.features import compute_rag
import h5py
from scipy.ndimage import binary_fill_holes
from cv2 import fitEllipse
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_conts(cont, shape, tolerance):
    """Helper to find a good setting for <tolerance>"""
    cont_image = np.zeros(shape)
    approx_image = np.zeros(shape)
    rr, cc = polygon_perimeter(cont[:, 0], cont[:, 1])
    cont_image[rr, cc] = 1
    poly_approx = approximate_polygon(cont, tolerance=tolerance)
    rra, cca = polygon_perimeter(poly_approx[:, 0], poly_approx[:, 1])
    approx_image[rra, cca] = 1
    plt.imshow(cont_image)
    plt.show()
    plt.imshow(approx_image)
    plt.show()

class CirclesRewards(RewardFunctionAbc):

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, prediction_segmentation, superpixel_segmentation, dir_edges, edge_score, sp_cmrads, actions,
                 *args, **kwargs):
        dev = prediction_segmentation.device
        return_scores = []
        exp_factor = 2

        for single_pred, single_sp_seg, s_dir_edges, s_actions, s_sp_cmrads in zip(prediction_segmentation,
                                                                                   superpixel_segmentation,
                                                                                   dir_edges, actions, sp_cmrads):
            # assert single_sp_seg.max() == s_dir_edges.max()
            # assert s_dir_edges.shape[1] > 60
            # assert single_sp_seg.max() > 20

            scores = torch.zeros(int((single_sp_seg.max()) + 1, ), device=dev)
            if single_pred.max() == 0:  # image is empty
                if edge_score:
                    edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]
                    edge_scores = scores[edges].max(dim=0).values
                    return_scores.append(edge_scores)
                else:
                    return_scores.append(scores)
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

            # get shape descriptors for potential_objects and get a shape_score by comparing to self.descriptors
            good_obj_cnt = 0
            for object, obj_id, sp_ids in zip(potential_objects, potential_object_ids, object_sp_ids):
                try:
                    contours = find_contours(object.cpu().numpy(), level=0)
                    if len(contours) == 0:
                        raise Exception()
                except:
                    scores[sp_ids] = 0.0
                    continue

                if len(contours) > 1:
                    scores[sp_ids] = 0.0
                    continue
                contour = np.array(contours[0], dtype=np.float)
                if (contour[0] == contour[-1]).sum() != 2:
                    continue
                cm = contour.mean(axis=0)
                dsts = np.linalg.norm(contour - cm, axis=1)
                dsts -= dsts.min()
                max = dsts.max()
                if max > 1:
                    dsts /= max
                score = 1 - (np.std(dsts) * 2)

                if score > 0.85:
                    good_obj_cnt += 1
                scores[sp_ids] += score.item()

            # score = 0.5 * (good_obj_cnt / 20) + 0.5 * (1 / len(bg_object_ids))
            score = 1 / len(bg_object_ids)
            score = np.exp((score * exp_factor)) / np.exp(np.array([exp_factor]))
            scores[bg_sp_ids] = score.item()
            scores[false_sp_ids] = 0.0
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                print(Warning("NaN or inf in scores this should not happen"))
                sys.stdout.flush()
                assert False
            if edge_score:
                edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]
                edge_scores = scores[edges].max(dim=0).values
                return_scores.append(edge_scores)
            else:
                return_scores.append(scores)

        return_scores = torch.cat(return_scores)
        return return_scores


class HoughCirclesRewards(RewardFunctionAbc):

    def __init__(self, *args, **kwargs):
        self.max_p = torch.nn.MaxPool2d(3, padding=1, stride=1)
        self.circle_thresh = 0.4
        self.range_rad = [10, 20]
        self.range_num = [20, 20]

    def __call__(self, prediction_segmentation, superpixel_segmentation, dir_edges, edge_score, sp_cmrads, actions,
                 *args, **kwargs):
        dev = prediction_segmentation.device
        return_scores = []
        exp_factor = 2

        for single_pred, single_sp_seg, s_dir_edges, s_actions, s_sp_cmrads in zip(prediction_segmentation,
                                                                                   superpixel_segmentation,
                                                                                   dir_edges, actions, sp_cmrads):
            # assert single_sp_seg.max() == s_dir_edges.max()
            # assert s_dir_edges.shape[1] > 60
            # assert single_sp_seg.max() > 20

            scores = torch.zeros(int((single_sp_seg.max()) + 1, ), device=dev)
            if single_pred.max() == 0:  # image is empty
                if edge_score:
                    edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]
                    edge_scores = scores[edges].max(dim=0).values
                    return_scores.append(edge_scores)
                else:
                    return_scores.append(scores)
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

                # plt.imshow(single_pred.cpu(), cmap=random_label_cmap(), interpolation="none"); plt.show()
                # ex = torch.zeros_like(single_pred).cpu()
                # for circle_idx in circle_idxs:
                #     ex[circle_idx[0], circle_idx[1]] = 1
                # plt.imshow(ex);plt.show()

                circle_sps = [torch.unique(single_sp_seg[circle_idx[0], circle_idx[1]]).long() for circle_idx in circle_idxs]
                obj_ids = [torch.unique(single_pred[circle_idx[0], circle_idx[1]]) for circle_idx in circle_idxs]

                for circle_sp, val, obj_id in zip(circle_sps, accums, obj_ids):
                    val = (val - self.circle_thresh) / (1 - self.circle_thresh)
                    hough_score = torch.sigmoid(torch.tensor([8 * (val - 0.5)])).item()
                    num_obj_score = 1 / max(len(obj_id), 1)
                    if num_obj_score == 1 and obj_id[0] in potential_object_ids:
                        good_obj_cnt += 1
                    scores[circle_sp] = 0.7 * hough_score + 0.3 * num_obj_score

            score = 1.0 * (good_obj_cnt / 15) * int(good_obj_cnt > 5) + 0.0 * (1 / len(bg_object_ids))
            # score = 1 / len(bg_object_ids)
            score = np.exp((score * exp_factor)) / np.exp(np.array([exp_factor]))
            scores[bg_sp_ids] = score.item()
            scores[false_sp_ids] = 0.0
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                print(Warning("NaN or inf in scores this should not happen"))
                sys.stdout.flush()
                assert False
            if edge_score:
                edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]
                edge_scores = scores[edges].max(dim=0).values
                return_scores.append(edge_scores)
            else:
                return_scores.append(scores)

        return_scores = torch.cat(return_scores)
        return return_scores

if __name__ == "__main__":
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from utils.general import multicut_from_probas, calculate_gt_edge_costs
    from glob import glob
    import os


    label_cm = random_label_cmap(zeroth=1.0)
    label_cm.set_bad(alpha=0)
    edge_cmap = cm.get_cmap(name="cool")
    edge_cmap.set_bad(alpha=0)
    dev = "cuda:0"
    # get a few images and extract some gt objects used ase shape descriptors that we want to compare against
    dir = "/g/kreshuk/hilt/projects/data/color_circles/train"
    fnames_pix = sorted(glob(os.path.join(dir, 'pix_data/*.h5')))
    fnames_graph = sorted(glob(os.path.join(dir, 'graph_data/*.h5')))

    for i in range(1):
        g_file = h5py.File(fnames_graph[i], 'r')
        pix_file = h5py.File(fnames_pix[i], 'r')
        superpixel_seg = g_file['node_labeling'][:]
        gt_seg = pix_file['gt'][:]
        superpixel_seg = torch.from_numpy(superpixel_seg.astype(np.int64)).to(dev)
        gt_seg = torch.from_numpy(gt_seg.astype(np.int64)).to(dev)

        probas = g_file['edge_feat'][:, 0]  # take initial edge features as weights

        # make sure probas are probas and get a sample prediction
        probas -= probas.min()
        probas /= (probas.max() + 1e-6)
        pred_seg = multicut_from_probas(superpixel_seg.cpu().numpy(), g_file['edges'][:].T, probas)
        pred_seg = torch.from_numpy(pred_seg.astype(np.int64)).to(dev)

        # relabel to consecutive integers:
        mask = gt_seg[None] == torch.unique(gt_seg)[:, None, None]
        gt_seg = (mask * (torch.arange(len(torch.unique(gt_seg)), device=dev)[:, None, None] + 1)).sum(0) - 1
        mask = superpixel_seg[None] == torch.unique(superpixel_seg)[:, None, None]
        superpixel_seg = (mask * (torch.arange(len(torch.unique(superpixel_seg)), device=dev)[:, None, None] + 1)).sum(
            0) - 1
        mask = pred_seg[None] == torch.unique(pred_seg)[:, None, None]
        pred_seg = (mask * (torch.arange(len(torch.unique(pred_seg)), device=dev)[:, None, None] + 1)).sum(0) - 1
        # assert the segmentations are consecutive integers
        assert pred_seg.max() == len(torch.unique(pred_seg)) - 1
        assert superpixel_seg.max() == len(torch.unique(superpixel_seg)) - 1

        edges = torch.from_numpy(compute_rag(superpixel_seg.cpu().numpy()).uvIds().astype(np.long)).T.to(dev)
        dir_edges = torch.stack((torch.cat((edges[0], edges[1])), torch.cat((edges[1], edges[0]))))

        gt_edges = calculate_gt_edge_costs(edges.T, superpixel_seg.squeeze(), gt_seg.squeeze(), thresh=0.5)

        mc_seg_old = None
        for itr in range(10):
            actions = gt_edges + torch.randn_like(gt_edges) * (itr / 10)
            # actions = torch.randn_like(gt_edges)
            # actions = torch.zeros_like(gt_edges)
            # actions[1:4] = 1.0
            actions -= actions.min()
            actions /= actions.max()

            mc_seg = multicut_from_probas(superpixel_seg.cpu().numpy(), edges.T.cpu().numpy(), actions)
            mc_seg = torch.from_numpy(mc_seg.astype(np.int64)).to(dev)

            # relabel to consecutive integers:
            mask = mc_seg[None] == torch.unique(mc_seg)[:, None, None]
            mc_seg = (mask * (torch.arange(len(torch.unique(mc_seg)), device=dev)[:, None, None] + 1)).sum(0) - 1
            # add batch dimension
            pred_seg = pred_seg[None]
            gt_seg = gt_seg[None]
            mc_seg = mc_seg[None]

            f = HoughCirclesRewards()
            # f.get_gaussians(.3, 0.16, .1, 0.2, 0.3, .2)
            # plt.imshow(pix_file['raw'][:]);plt.show()
            # rewards2 = f(gt_seg.long(), superpixel_seg.long(), dir_edges=[dir_edges], res=100)
            edge_angles, sp_feat, sp_rads = get_angles_smass_in_rag(edges, superpixel_seg.long())
            edge_rewards = f(mc_seg.long(), superpixel_seg[None].long(), dir_edges=[dir_edges], res=50, edge_score=True,
                             sp_cmrads=[sp_rads], actions=[actions])

            fig, ax = plt.subplots(2, 2)
            frame_rew, scores_rew, bnd_mask = get_colored_edges_in_sseg(superpixel_seg[None].float(), edges, edge_rewards)
            frame_act, scores_act, _ = get_colored_edges_in_sseg(superpixel_seg[None].float(), edges, 1 - actions.squeeze())

            bnd_mask = torch.from_numpy(dilation(bnd_mask.cpu().numpy()))

            # frame_rew = np.stack([dilation(frame_rew.cpu().numpy()[..., i]) for i in range(3)], -1)
            # frame_act = np.stack([dilation(frame_act.cpu().numpy()[..., i]) for i in range(3)], -1)
            # scores_rew = np.stack([dilation(scores_rew.cpu().numpy()[..., i]) for i in range(3)], -1)
            # scores_act = np.stack([dilation(scores_act.cpu().numpy()[..., i]) for i in range(3)], -1)

            ax[1, 0].imshow(mc_seg.cpu().squeeze(), cmap=label_cm, interpolation="none")
            ax[1, 0].set_title("mc_seg")
            ax[1, 0].axis('off')
            ax[1, 1].imshow(superpixel_seg.cpu().squeeze(), cmap=label_cm, interpolation="none")
            ax[1, 1].set_title("sp_seg")
            ax[1, 1].axis('off')

            mc_seg = mc_seg.squeeze().float()
            mc_seg[bnd_mask] = np.nan
            ax[0, 0].imshow(frame_rew.cpu().numpy(), interpolation="none")
            ax[0, 0].imshow(mc_seg.cpu(), cmap=label_cm, alpha=0.8, interpolation="none")
            ax[0, 0].set_title("rewards 1:g, 0:r")
            ax[0, 0].axis('off')
            ax[0, 1].imshow(frame_act.cpu().numpy(), interpolation="none")
            ax[0, 1].imshow(mc_seg.cpu(), cmap=label_cm, alpha=0.8, interpolation="none")
            ax[0, 1].set_title("actions 0:g, 1:r")
            ax[0, 1].axis('off')

            # plt.savefig("/g/kreshuk/hilt/projects/RLForSeg/data/test.png", bbox_inches='tight')
            plt.show()

        # rewards1 = f(pred_seg.long(), superpixel_seg.long(), dir_edges=[dir_edges], res=100)
    a = 1
