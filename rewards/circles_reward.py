import matplotlib
# matplotlib.use('TkAgg')
from rewards.reward_abc import RewardFunctionAbc
from skimage.measure import approximate_polygon, find_contours
from skimage.morphology import dilation
from skimage.draw import polygon_perimeter, line
from utils.polygon_2d import Polygon2d
from utils.general import random_label_cmap, get_colored_edges_in_sseg, sync_segmentations
from utils.graphs import get_angles_smass_in_rag
import torch
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
        exp_factor = 6

        for single_pred, single_sp_seg, s_dir_edges, s_actions, s_sp_cmrads in zip(prediction_segmentation,
                                                                                   superpixel_segmentation,
                                                                                   dir_edges, actions, sp_cmrads):
            scores = torch.zeros(int((single_sp_seg.max()) + 1, ), device=dev)
            if single_pred.max() == 0:  # image is empty
                return_scores.append(scores)
                continue
            # get one-hot representation
            one_hot = torch.zeros((int(single_pred.max()) + 1,) + single_pred.size(), device=dev, dtype=torch.long) \
                .scatter_(0, single_pred[None], 1)

            # need masses to determine what objects can be considered background
            label_masses = one_hot.flatten(1).sum(-1)
            bg1_mask = torch.zeros_like(label_masses).bool()
            bg1_id = label_masses.argmax()
            bg1_mass = label_masses[bg1_id].item()
            bg1_mask[bg1_id] = True
            bg2_mask = torch.zeros_like(label_masses).bool()
            label_masses[bg1_id] = -1
            bg2_id = label_masses.argmax()
            label_masses[bg1_id] = bg1_mass
            bg2_mask[bg2_id] = True
            # get the objects that are torching the patch boarder as for them we cannot compute a relieable sim shape_score
            false_obj_mask = label_masses < 20
            false_obj_mask[bg1_id] = False
            false_obj_mask[bg2_id] = False
            # everything else are potential objects
            potenial_obj_mask = (false_obj_mask == False)
            potenial_obj_mask[bg1_id] = False
            potenial_obj_mask[bg2_id] = False
            potential_object_ids = torch.nonzero(potenial_obj_mask).squeeze(1)  # object label IDs

            objects = one_hot[potential_object_ids]  # get object masks
            false_obj_sp_ids = torch.unique((single_sp_seg[None] + 1) * one_hot[false_obj_mask])[1:] - 1
            object_sp_ids = [torch.unique((single_sp_seg[None] + 1) * obj)[1:] - 1 for obj in objects]

            # get shape descriptors for objects and get a shape_score by comparing to self.descriptors

            for object, obj_id, sp_ids in zip(objects, potential_object_ids, object_sp_ids):
                try:
                    contour = find_contours(object.cpu().numpy(), level=0)
                    if len(contour) > 1:
                        continue
                    contour = contour[0]
                except:
                    continue

                poly_chain = torch.from_numpy(approximate_polygon(contour, tolerance=0.0)).to(dev)
                cm = poly_chain.mean(dim=0).cpu().numpy()
                dsts = np.linalg.norm(poly_chain.cpu().numpy() - cm, axis=1)
                dsts -= dsts.min()
                if dsts.max() > 1:
                    dsts /= dsts.max()
                score = 1 - (np.std(dsts) * 2)

                # score = (score * exp_factor).exp() / (torch.ones_like(score) * exp_factor).exp()
                scores[sp_ids] = score

            scores[false_obj_sp_ids] = 0.0
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                print(Warning("NaN or inf in scores this should not happen"))
            if edge_score:
                edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]
                edge_scores = scores[edges].max(dim=0).values
                return_scores.append(edge_scores)
            else:
                return_scores.append(scores)

        return_scores = torch.cat(return_scores)
        return return_scores
