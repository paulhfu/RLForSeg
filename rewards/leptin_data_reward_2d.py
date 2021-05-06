import matplotlib
# matplotlib.use('TkAgg')
import skimage.draw

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

def show_circle(raw, center, perimeters):
    raw /= raw.max()
    raw = torch.stack((raw, raw, raw), -1).float().numpy()
    for perimeter in perimeters:
        rr, cc = skimage.draw.circle_perimeter(center[0], center[1], perimeter, shape=raw.shape)
        raw[rr, cc] = np.array([1., 0., 0.])
    plt.imshow(raw)
    plt.show()


class LeptinDataRotatedRectRewards(RewardFunctionAbc):

    def __init__(self, device="cuda:0", *args, **kwargs):
        source_file_wtsd = "/g/kreshuk/data/leptin/sourabh_data_v1/Segmentation_results_fused_tp_1_ch_0_Masked_WatershedBoundariesMergeTreeFilter_Out1.tif"
        source_file_wtsd = "/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/Masked_WatershedBoundariesMergeTreeFilter_Out1.h5"
        # wtsd = torch.from_numpy(np.array(imread(source_file_wtsd).astype(np.long))).to(device)
        wtsd = torch.from_numpy(h5py.File(source_file_wtsd, "r")["data"][:].astype(np.long)).to(device)
        slices = [0, 157, 316]
        label_1 = [1359, 886, 1240]
        label_2 = [1172, 748, 807]
        label_3 = [364, 1148, 1447]
        m1, m2, m3, m4, m5 = [], [], [], [], []
        self.outer_cntr_ds, self.inner_cntr_ds, self.celltype_1_ds, self.celltype_2_ds, self.celltype_3_ds = [], [], [], [], []
        for slc, l1, l2, l3 in zip(slices, label_1, label_2, label_3):
            bg = wtsd[:, slc, :] == 1
            bg_cnt = find_contours(bg.cpu().numpy(), level=0)
            cnt1 = bg_cnt[0] if bg_cnt[0].shape[0] > bg_cnt[1].shape[0] else bg_cnt[1]
            cnt2 = bg_cnt[1] if bg_cnt[0].shape[0] > bg_cnt[1].shape[0] else bg_cnt[0]
            for m, cnt in zip([m1, m2], [cnt1, cnt2]):
                mask = torch.zeros_like(wtsd[:, slc, :]).cpu()
                mask[np.round(cnt[:, 0]), np.round(cnt[:, 1])] = 1
                m.append(torch.from_numpy(binary_fill_holes(mask.long().cpu().numpy())).to(device).sum().item())

            mask = wtsd[:, slc, :] == l1
            m3.append(mask.long().sum().item())
            cnt3 = find_contours(mask.cpu().numpy(), level=0)[0]
            mask = wtsd[:, slc, :] == l2
            m4.append(mask.long().sum().item())
            cnt4 = find_contours(mask.cpu().numpy(), level=0)[0]
            mask = wtsd[:, slc, :] == l3
            m5.append(mask.long().sum().item())
            cnt5 = find_contours(mask.cpu().numpy(), level=0)[0]

            self.outer_cntr_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt1, tolerance=1.2)).to(device)))
            self.inner_cntr_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt2, tolerance=1.2)).to(device)))
            self.celltype_1_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt3, tolerance=1.2)).to(device)))
            self.celltype_2_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt4, tolerance=1.2)).to(device)))
            self.celltype_3_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt5, tolerance=1.2)).to(device)))

        self.masses = [np.array(m1).mean(), np.array(m2).mean(), np.array(m3 + m4 + m5).mean()]
        self.fg_shape_descriptors = self.celltype_1_ds + self.celltype_2_ds + self.celltype_3_ds
        # self.circle_center = np.array([390, 340])
        self.circle_rads = [210, 270, 100, 340]
        self.exact_circle_diameter = [345, 353, 603, 611]
        self.side_lens = [28, 125]
        self.problem_area = [95/275, 355/32]

    def __call__(self, prediction_segmentation, superpixel_segmentation, dir_edges, edge_score, sp_cmrads, actions,
                 sp_cms, sp_masses, *args, **kwargs):
        dev = prediction_segmentation.device
        return_scores = []
        exp_factor = 2

        for single_pred, single_sp_seg, s_dir_edges, s_actions, s_sp_cmrads, cms, masses in zip(prediction_segmentation,
                                                                                   superpixel_segmentation,
                                                                                   dir_edges, actions, sp_cmrads,
                                                                                   sp_cms, sp_masses):
            scores = torch.zeros(int((single_sp_seg.max()) + 1, ), device=dev)
            if single_pred.max() == 0:  # image is empty
                if edge_score:
                    edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]
                    edge_scores = scores[edges].max(dim=0).values
                    return_scores.append(edge_scores)
                else:
                    return_scores.append(scores)
                continue

            circle_center = cms.T[masses < 1000].mean(0).round().long().cpu().numpy()

            # get one-hot representation
            one_hot = torch.zeros((int(single_pred.max()) + 1,) + single_pred.size(), device=dev, dtype=torch.long) \
                .scatter_(0, single_pred[None], 1)
            label_masses = one_hot.flatten(1).sum(-1)
            meshgrid = torch.stack(torch.meshgrid(torch.arange(single_pred.shape[0], device=dev),
                                                  torch.arange(single_pred.shape[1], device=dev)))
            radii = (((one_hot[:, None].float() * meshgrid[None].float()) -
                     torch.from_numpy(circle_center).to(dev).float()[None, :, None, None]) ** 2).sum(1).sqrt()
            bg1_score = ((radii > self.circle_rads[-1]) * one_hot).flatten(1).sum(1)
            bg2_score = ((radii < self.circle_rads[-2]) * one_hot).flatten(1).sum(1)
            bg1_id = torch.argmax(bg1_score)
            bg2_id = torch.argmax(bg2_score)

            # need masses to determine what objects can be considered background
            bg1_mask = torch.zeros_like(label_masses).bool()
            # bg1_id = label_masses.argmax()
            # bg1_mass = label_masses[bg1_id].item()
            bg1_mask[bg1_id] = True
            bg2_mask = torch.zeros_like(label_masses).bool()
            # label_masses[bg1_id] = -1
            # bg2_id = label_masses.argmax()
            # label_masses[bg1_id] = bg1_mass
            bg2_mask[bg2_id] = True
            # get the objects that are torching the patch boarder as for them we cannot compute a relieable sim shape_score
            false_obj_mask = (label_masses < (self.masses[2] / 3)) | (label_masses > (self.masses[2] * 3))
            false_obj_mask[bg1_id] = False
            false_obj_mask[bg2_id] = False
            # everything else are potential objects
            potenial_obj_mask = (false_obj_mask == False)
            potenial_obj_mask[bg1_id] = False
            potenial_obj_mask[bg2_id] = False
            potential_object_ids = torch.nonzero(potenial_obj_mask).squeeze(1)  # object label IDs

            bg1 = one_hot[bg1_id]  # get background masks
            bg2 = one_hot[bg2_id]  # get background masks
            objects = one_hot[potential_object_ids]  # get object masks
            false_obj_sp_ids = torch.unique((single_sp_seg[None] + 1) * one_hot[false_obj_mask])[1:] - 1
            bg1_sp_ids = torch.unique((single_sp_seg[None] + 1) * bg1)[1:] - 1
            bg2_sp_ids = torch.unique((single_sp_seg[None] + 1) * bg2)[1:] - 1
            object_sp_ids = [torch.unique((single_sp_seg[None] + 1) * obj)[1:] - 1 for obj in objects]

            # get shape descriptors for objects and get a shape_score by comparing to self.descriptors

            for object, obj_id, sp_ids in zip(objects, potential_object_ids, object_sp_ids):
                try:
                    contour = find_contours(object.cpu().numpy(), level=0)
                    if len(contour) > 1:
                        continue
                    contour = contour[0]
                    if len(contour) < 4:
                        continue
                except:
                    continue

                poly_chain = torch.tensor(contour).to(dev)
                cm = poly_chain.mean(dim=0).cpu()
                pos = circle_center-cm.cpu().tolist()
                ang = abs(pos[1]) / abs(pos[0]) if (pos > 0).all() else None
                in_problem_area = False if ang is None else ang > self.problem_area[0] and ang < self.problem_area[1]
                position = ((cm - circle_center) ** 2).sum().sqrt()
                dt = abs(self.circle_rads[0] - position)
                position_score = 0.0
                if dt < 100:
                    position_score = 1

                rect = cv2.minAreaRect(contour.astype(np.int))
                rect = np.round(cv2.boxPoints(rect)).astype(np.int)
                dsts = [math.sqrt((rect[pt - 1][0] - rect[pt % 4][0]) ** 2 + (rect[pt - 1][1] - rect[pt % 4][1]) ** 2)
                        for pt in range(1, 5)]
                dsts = np.array(dsts)
                long_side_ind = dsts.argmax()
                long_side = dsts[long_side_ind]
                short_side = dsts.min()

                u = np.array([rect[long_side_ind], rect[(long_side_ind + 1) % 4]]) - circle_center
                u = u[0] - u[1]
                v = cm - circle_center
                u = u / (np.linalg.norm(u) + 1e-10)
                v = v / (np.linalg.norm(v) + 1e-10)
                orientation_score = abs(np.dot(u, v))

                shape_score = 0.0
                diffs, diffl = abs(self.side_lens[0] - short_side), abs(self.side_lens[1] - long_side)
                if not in_problem_area:
                    if diffs < (self.side_lens[0] / 4):
                        shape_score += 0.5
                    elif diffs < (self.side_lens[0] / 2):
                        shape_score += 0.3
                    elif diffs < self.side_lens[0]:
                        shape_score += 0.2

                    if diffl < (self.side_lens[1] / 6):
                        shape_score += 0.5
                    elif diffl < (self.side_lens[1] / 5):
                        shape_score += 0.3
                    elif diffl < (self.side_lens[1] / 3):
                        shape_score += 0.2
                    elif diffl < self.side_lens[1]:
                        shape_score += 0.1
                else:
                    orientation_score = 1.0
                    if diffs < (self.side_lens[0] / 1.5):
                        shape_score += 0.5
                    elif diffs < (self.side_lens[0] * 2):
                        shape_score += 0.3
                    elif diffs < self.side_lens[0] * 3:
                        shape_score += 0.2

                    if diffl < (self.side_lens[1] / 4):
                        shape_score += 0.5
                    elif diffl < (self.side_lens[1] / 2):
                        shape_score += 0.3
                    elif diffl < (self.side_lens[1] / 1):
                        shape_score += 0.2
                    elif diffl < self.side_lens[1]:
                        shape_score += 0.1

                score = torch.tensor([0.1 * position_score + 0.8 * shape_score + 0.2 * orientation_score], device=dev)
                score = (score * exp_factor).exp() / (torch.ones_like(score) * exp_factor).exp()

                if shape_score < 0.2 or orientation_score < 0.3:
                    scores[sp_ids] = 0
                else:
                    scores[sp_ids] = score.item()

            bg1_shape_score, bg2_shape_score = 0, 0
            # penalize size variation
            diff1 = (label_masses[bg1_id] - self.masses[0]).abs()
            diff2 = (label_masses[bg2_id] - self.masses[1]).abs()
            if diff1 > (self.masses[0] / 4):
                scores[bg1_sp_ids] = 0.0
            else:
                try:
                    contour = find_contours(bg1.cpu().numpy(), level=0)
                    # if len(contour) > 1:
                    #     raise Exception()
                    contour = contour[np.array([len(cnt) for cnt in contour]).argmax()]
                    contour = torch.from_numpy(contour).to(dev)
                    # dsts = ((contour.cpu() - circle_center)**2).sum(1).sqrt()
                    cm = contour.mean(0)
                    if (cm.cpu() - circle_center).abs().sum() > 100:
                        scores[bg1_sp_ids] = 0.0
                    elif contour.shape[0] <= 8:
                        scores[bg1_sp_ids] = 0.0
                    # elif ((dsts < self.circle_rads[1]).sum() / len(dsts)) > 0.3:
                    #     scores[bg1_sp_ids] = 0.0
                    else:
                        bg1_shape_score = 0.0
                        principal_ax = fitEllipse(contour.long().cpu().numpy())[1]
                        diff1 = abs(max(principal_ax) - self.exact_circle_diameter[3])
                        diff2 = abs(min(principal_ax) - self.exact_circle_diameter[2])

                        if diff2 < 15:
                            bg1_shape_score += 0.5
                        elif diff2 < 20:
                            bg1_shape_score += 0.4
                        elif diff2 < 30:
                            bg1_shape_score += 0.3
                        elif diff2 < 40:
                            bg1_shape_score += 0.2
                        elif diff2 < 50:
                            bg1_shape_score += 0.1

                        if diff1 < 15:
                            bg1_shape_score += 0.5
                        elif diff1 < 20:
                            bg1_shape_score += 0.4
                        elif diff1 < 30:
                            bg1_shape_score += 0.3
                        elif diff1 < 40:
                            bg1_shape_score += 0.2
                        elif diff1 < 50:
                            bg1_shape_score += 0.1

                        bg1_shape_score = torch.tensor(bg1_shape_score)
                        bg1_shape_score = (bg1_shape_score * exp_factor).exp() / (torch.ones_like(bg1_shape_score) * exp_factor).exp()
                        scores[bg1_sp_ids] = bg1_shape_score.item()

                        scores[bg1_sp_ids][s_sp_cmrads[bg1_sp_ids] < self.circle_rads[1]] = 0

                        # polygon = Polygon2d(poly_chain)
                        # shape_dist_scores = torch.tensor([des.distance(polygon, 200) for des in self.outer_cntr_ds], device=dev)
                        # # shape_score = (torch.sigmoid((((1 - shape_dist_scores.min()) * 6.5).exp() / torch.tensor([6.5], device=dev).exp()) * 6 - 3) * 1.2597) - 0.2
                        # bg1_shape_score = torch.tensor([1 - shape_dist_scores.min()], device=dev)

                        # bg1_shape_score = (bg1_shape_score * exp_factor).exp() / (
                        #             torch.ones_like(bg1_shape_score) * exp_factor).exp()

                        # rads = np.linalg.norm(contour.cpu().numpy() - circle_center[np.newaxis], axis=1)
                        # dists = abs(rads.mean() - rads)
                        # bg1_rad = rads[dists < 10].mean() - 5
                except Exception as e:
                    print(e)
            if diff2 > (self.masses[1] / 4):
                scores[bg2_sp_ids] = 0.0
            else:
                try:
                    contour = find_contours(bg2.cpu().numpy(), level=0)
                    # if len(contour) > 1:
                    #     raise Exception
                    contour = contour[np.array([len(cnt) for cnt in contour]).argmax()]
                    contour = torch.from_numpy(contour).to(dev)
                    # dsts = ((contour.cpu() - circle_center) ** 2).sum(1).sqrt()
                    cm = contour.mean(0)
                    if (cm.cpu() - circle_center).abs().sum() > 150:
                        scores[bg2_sp_ids] = 0.0
                    elif contour.shape[0] <= 8:
                        scores[bg2_sp_ids] = 0.0
                    # elif ((dsts > self.circle_rads[0]).sum() / len(contour)) > 0.3:
                    #     scores[bg2_sp_ids] = 0.0
                    else:

                        bg2_shape_score = 0.0
                        principal_ax = fitEllipse(contour.long().cpu().numpy())[1]
                        diff1 = abs(max(principal_ax) - self.exact_circle_diameter[1])
                        diff2 = abs(min(principal_ax) - self.exact_circle_diameter[0])

                        # plt.imshow(bg2.cpu()[..., None] * 200 + cv2.ellipse(np.zeros(shape=[741, 692, 3], dtype=np.uint8), fitEllipse(contour.long().cpu().numpy()), (23, 184, 80), 3))
                        # plt.show()

                        if diff2 < 15:
                            bg2_shape_score += 0.5
                        elif diff2 < 20:
                            bg2_shape_score += 0.4
                        elif diff2 < 30:
                            bg2_shape_score += 0.3
                        elif diff2 < 40:
                            bg2_shape_score += 0.2
                        elif diff2 < 50:
                            bg2_shape_score += 0.1

                        if diff1 < 15:
                            bg2_shape_score += 0.5
                        elif diff1 < 20:
                            bg2_shape_score += 0.4
                        elif diff1 < 30:
                            bg2_shape_score += 0.3
                        elif diff1 < 40:
                            bg2_shape_score += 0.2
                        elif diff1 < 50:
                            bg2_shape_score += 0.1

                        bg2_shape_score = torch.tensor(bg2_shape_score)
                        bg2_shape_score = (bg2_shape_score * exp_factor).exp() / (torch.ones_like(bg2_shape_score) * exp_factor).exp()
                        scores[bg2_sp_ids] = bg2_shape_score.item()

                        scores[bg2_sp_ids][s_sp_cmrads[bg2_sp_ids] > self.circle_rads[0]] = 0

                        # polygon = Polygon2d(poly_chain)
                        # shape_dist_scores = torch.tensor([des.distance(polygon, 200) for des in self.inner_cntr_ds],
                        #                                  device=dev)
                        # # shape_score = (torch.sigmoid((((1 - shape_dist_scores.min()) * 6.5).exp() / torch.tensor([6.5], device=dev).exp()) * 6 - 3) * 1.2597) - 0.2
                        # bg2_shape_score = torch.tensor([1 - shape_dist_scores.min()], device=dev)
                        # bg2_shape_score = (bg2_shape_score * exp_factor).exp() / (
                        #             torch.ones_like(bg2_shape_score) * exp_factor).exp()

                        # rads = np.linalg.norm(contour.cpu().numpy() - circle_center[np.newaxis], axis=1)
                        # dists = abs(rads.mean() - rads)
                        # bg2_rad = rads[dists < 10].mean() + 5
                except Exception as e:
                    print(e)

            scores[false_obj_sp_ids] = 0.0
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                print(Warning("NaN or inf in scores this should not happen"))
            if edge_score:
                # .02, 0.1, .1, 0.2, 0.25, .3
                # .02, 0.13, .12, 0.2, 0.4, .3
                s1 = .02
                s2 = .1
                s3 = .1
                w1 = .2
                w2 = .25
                w3 = .3
                n = math.sqrt((single_sp_seg.shape[0] / 2) ** 2 + (single_sp_seg.shape[1] / 2) ** 2)
                edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]
                edge_scores = scores[edges].max(dim=0).values
                # identify wrong edges
                edge_cmrads = s_sp_cmrads[edges]

                bg_sp_1 = torch.arange(len(s_sp_cmrads), device=dev)[s_sp_cmrads > self.circle_rads[1]]
                bg_sp_2 = torch.arange(len(s_sp_cmrads), device=dev)[s_sp_cmrads < self.circle_rads[0]]

                edge_mask_1 = (edges[None] == bg_sp_1[:, None, None]).sum(0).sum(0) == 2
                edge_mask_2 = (edges[None] == bg_sp_2[:, None, None]).sum(0).sum(0) == 2

                mdst1 = self.circle_rads[3] - edge_cmrads[:, edge_mask_1].mean(0)
                dst1 = torch.clamp(mdst1 / self.circle_rads[3], min=0)
                dst2 = (edge_cmrads[:, edge_mask_2].mean(0)) / n
                dst3 = (edge_cmrads[:, edge_mask_1].mean(0) - ((self.circle_rads[1] + self.circle_rads[0]) / 2)) / n
                dst4 = (edge_cmrads[:, edge_mask_2].mean(0) - ((self.circle_rads[1] + self.circle_rads[0]) / 2)) / n

                bg_prob1 = torch.exp(-dst1 ** 2 / (2 * s1 ** 2)) / (math.sqrt(2 * np.pi) * s1) * w1
                bg_prob2 = torch.exp(-dst2 ** 2 / (2 * s2 ** 2)) / (math.sqrt(2 * np.pi) * s2) * w2
                fg_prob1 = torch.exp(-dst3 ** 2 / (2 * s3 ** 2)) / (math.sqrt(2 * np.pi) * s3) * w3
                fg_prob2 = torch.exp(-dst4 ** 2 / (2 * s3 ** 2)) / (math.sqrt(2 * np.pi) * s3) * w3

                # bg_prob1 = torch.clamp(bg_prob1, max=1)
                # bg_prob2 = torch.clamp(bg_prob2, max=1)
                # fg_prob1 = torch.clamp(fg_prob1, max=1)
                # fg_prob2 = torch.clamp(fg_prob2, max=1)
                #
                weight1, weight2 = fg_prob1 + bg_prob1, fg_prob2 + bg_prob2
                fg_prob1, bg_prob1 = fg_prob1 / weight1, bg_prob1 / weight1
                fg_prob2, bg_prob2 = fg_prob2 / weight2, bg_prob2 / weight2

                edge_scores[edge_mask_1] = fg_prob1 * edge_scores[edge_mask_1] + (1 - s_actions[edge_mask_1]) * bg_prob1
                edge_scores[edge_mask_2] = fg_prob2 * edge_scores[edge_mask_2] + (1 - s_actions[edge_mask_2]) * bg_prob2

                edge_mask = (edges[None] == bg2_sp_ids[:, None, None]).sum(0).sum(0) == 2
                unmerge_edges = (edge_cmrads[:, edge_mask] < self.circle_rads[1]).sum(0) == 1
                edge_scores[edge_mask][unmerge_edges] = s_actions[edge_mask][unmerge_edges]

                edge_mask = (edges[None] == bg2_sp_ids[:, None, None]).sum(0).sum(0) == 2
                unmerge_edges = (edge_cmrads[:, edge_mask] > self.circle_rads[0]).sum(0) == 1
                edge_scores[edge_mask][unmerge_edges] = s_actions[edge_mask][unmerge_edges]

                return_scores.append(edge_scores)
            else:
                return_scores.append(scores)

        return_scores = torch.cat(return_scores)
        return return_scores

    def get_gaussians(self, s1, s2, s3, w1, w2, w3):
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        xx, yy = np.meshgrid(np.arange(0, 749), np.arange(0, 692))
        pix_rads = np.sqrt((xx - self.circle_center[1])**2 + (yy - self.circle_center[0])**2)
        bg_2_mask = pix_rads > self.circle_rads[1]
        bg_1_mask = pix_rads < self.circle_rads[0]
        fg_mask = (pix_rads > self.circle_rads[2]) & (pix_rads < self.circle_rads[3])

        # normalizer3 = abs(np.linalg.norm(self.circle_center) - ((self.circle_rads[2] + self.circle_rads[3]) / 2)) / 0.5
        # normalizer2 = abs(np.linalg.norm(self.circle_center) - self.circle_rads[1]) / 0.5
        # normalizer1 = self.circle_rads[0] / 0.5

        # n3 = 30
        # n2 = 60
        # n1 = 40

        dst2 = np.zeros_like(pix_rads)
        dst1 = np.zeros_like(pix_rads)
        dst3 = np.zeros_like(pix_rads)
        bg_prob2 = np.zeros_like(pix_rads)
        bg_prob1 = np.zeros_like(pix_rads)
        fg_prob = np.zeros_like(pix_rads)
        n =math.sqrt((749/2)**2 + (692/2)**2)

        mdst1 = self.circle_rads[3] - pix_rads
        dst1 = np.clip(mdst1 / self.circle_rads[3], a_min=0, a_max=np.inf)
        dst2 = (pix_rads / n)
        dst3 = ((pix_rads - ((self.circle_rads[0] + self.circle_rads[1]) / 2)) / n)

        bg_prob1 = np.exp(-dst1 ** 2 / (2 * s1 ** 2)) / (math.sqrt(2 * np.pi) * s1) * w1
        bg_prob1 = np.clip(bg_prob1, a_min=0, a_max=1)
        # plt.imshow(bg_prob1);
        # plt.show()
        bg_prob2 = np.exp(-dst2 ** 2 / (2 * s2 ** 2)) / (math.sqrt(2 * np.pi) * s2) * w2
        # plt.imshow(bg_prob2);
        # plt.show()
        fg_prob = np.exp(-dst3 ** 2 / (2 * s3 ** 2)) / (math.sqrt(2 * np.pi) * s3) * w3
        # plt.imshow(fg_prob);
        # plt.show()

        bg_prob1 = np.clip(bg_prob1, a_max=1, a_min=0)
        bg_prob2 = np.clip(bg_prob2, a_max=1, a_min=0)
        fg_prob = np.clip(fg_prob, a_max=1, a_min=0)

        zz = bg_prob2 + bg_prob1 + fg_prob
        print(f"z(210):{zz[375, self.circle_center[1] - 210]}  z(260):{zz[375, self.circle_center[1] - 260]}  fgn(210): {fg_prob[375,self.circle_center[1] -  210]}  fgn(260): {fg_prob[375, self.circle_center[1] - 260]}  bgn1(260): {bg_prob1[375, self.circle_center[1] - 260]}  bgn2(210): {bg_prob2[375, self.circle_center[1] - 210]}")

        # fig = plt.figure(0)
        # ax = Axes3D(fig)
        # ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap=cm.magma, linewidth=0, antialiased=False)
        # plt.show()
        plt.plot(zz[375])
        plt.show()
        # fig = plt.figure(0)
        # ax = Axes3D(fig)
        # ax.plot_surface(xx, yy, bg_prob1, rstride=1, cstride=1, cmap=cm.magma, linewidth=0, antialiased=False)
        # plt.show()
        # fig = plt.figure(0)
        # ax = Axes3D(fig)
        # ax.plot_surface(xx, yy, fg_prob, rstride=1, cstride=1, cmap=cm.magma, linewidth=0, antialiased=False)
        # plt.show()
        a = 1


class LeptinDataReward2DTurningWithEllipses(RewardFunctionAbc):

    def __init__(self, device="cuda:0", *args, **kwargs):
        source_file_wtsd = "/g/kreshuk/data/leptin/sourabh_data_v1/Segmentation_results_fused_tp_1_ch_0_Masked_WatershedBoundariesMergeTreeFilter_Out1.tif"
        source_file_wtsd = "/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/Masked_WatershedBoundariesMergeTreeFilter_Out1.h5"
        # wtsd = torch.from_numpy(np.array(imread(source_file_wtsd).astype(np.long))).to(device)
        wtsd = torch.from_numpy(h5py.File(source_file_wtsd, "r")["data"][:].astype(np.long)).to(device)
        slices = [0, 157, 316]
        label_1 = [1359, 886, 1240]
        label_2 = [1172, 748, 807]
        label_3 = [364, 1148, 1447]
        m1, m2, m3, m4, m5 = [], [], [], [], []
        self.outer_cntr_ds, self.inner_cntr_ds, self.celltype_1_ds, self.celltype_2_ds, self.celltype_3_ds = [], [], [], [], []
        for slc, l1, l2, l3 in zip(slices, label_1, label_2, label_3):
            bg = wtsd[:, slc, :] == 1
            bg_cnt = find_contours(bg.cpu().numpy(), level=0)
            cnt1 = bg_cnt[0] if bg_cnt[0].shape[0] > bg_cnt[1].shape[0] else bg_cnt[1]
            cnt2 = bg_cnt[1] if bg_cnt[0].shape[0] > bg_cnt[1].shape[0] else bg_cnt[0]
            for m, cnt in zip([m1, m2], [cnt1, cnt2]):
                mask = torch.zeros_like(wtsd[:, slc, :]).cpu()
                mask[np.round(cnt[:, 0]), np.round(cnt[:, 1])] = 1
                m.append(torch.from_numpy(binary_fill_holes(mask.long().cpu().numpy())).to(device).sum().item())

            mask = wtsd[:, slc, :] == l1
            m3.append(mask.long().sum().item())
            cnt3 = find_contours(mask.cpu().numpy(), level=0)[0]
            mask = wtsd[:, slc, :] == l2
            m4.append(mask.long().sum().item())
            cnt4 = find_contours(mask.cpu().numpy(), level=0)[0]
            mask = wtsd[:, slc, :] == l3
            m5.append(mask.long().sum().item())
            cnt5 = find_contours(mask.cpu().numpy(), level=0)[0]

            self.outer_cntr_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt1, tolerance=1.2)).to(device)))
            self.inner_cntr_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt2, tolerance=1.2)).to(device)))
            self.celltype_1_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt3, tolerance=1.2)).to(device)))
            self.celltype_2_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt4, tolerance=1.2)).to(device)))
            self.celltype_3_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt5, tolerance=1.2)).to(device)))

        self.masses = [np.array(m1).mean(), np.array(m2).mean(), np.array(m3 + m4 + m5).mean()]
        self.fg_shape_descriptors = self.celltype_1_ds + self.celltype_2_ds + self.celltype_3_ds
        self.circle_center = [390, 340]
        self.circle_rads = [210, 280, 320, 120]

    def __call__(self, prediction_segmentation, superpixel_segmentation, res, dir_edges, edge_score, sp_cmrads, actions,
                 *args, **kwargs):
        dev = prediction_segmentation.device
        return_scores = []

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
            false_obj_mask = label_masses < 4
            false_obj_mask[bg1_id] = False
            false_obj_mask[bg2_id] = False
            # everything else are potential objects
            potenial_obj_mask = (false_obj_mask == False)
            potenial_obj_mask[bg1_id] = False
            potenial_obj_mask[bg2_id] = False
            potential_object_ids = torch.nonzero(potenial_obj_mask).squeeze(1)  # object label IDs

            bg1 = one_hot[bg1_id]  # get background masks
            bg2 = one_hot[bg2_id]  # get background masks
            objects = one_hot[potential_object_ids]  # get object masks
            false_obj_sp_ids = torch.unique((single_sp_seg[None] + 1) * one_hot[false_obj_mask])[1:] - 1
            bg1_sp_ids = torch.unique((single_sp_seg[None] + 1) * bg1)[
                         1:] - 1  # mask out the covered superpixels (need to add 1 because the single_sp_seg start from 0)
            bg2_sp_ids = torch.unique((single_sp_seg[None] + 1) * bg2)[
                         1:] - 1  # mask out the covered superpixels (need to add 1 because the single_sp_seg start from 0)
            object_sp_ids = [torch.unique((single_sp_seg[None] + 1) * obj)[1:] - 1 for obj in objects]

            # get shape descriptors for objects and get a shape_score by comparing to self.descriptors

            for object, obj_id, sp_ids in zip(objects, potential_object_ids, object_sp_ids):
                diff = label_masses[obj_id] - self.masses[2]
                if diff > 0 and label_masses[obj_id] > self.masses[2] * 3 or diff < 0 and label_masses[obj_id] < \
                        self.masses[2] / 3:
                    continue
                try:
                    contour = find_contours(object.cpu().numpy(), level=0)
                    if len(contour) > 1:
                        continue
                    contour = contour[0]
                except:
                    continue
                poly_chain = torch.from_numpy(approximate_polygon(contour, tolerance=0.0)).to(dev)
                if poly_chain.shape[0] <= 3:
                    continue
                polygon = Polygon2d(poly_chain)

                cm = poly_chain.mean(dim=0)
                position_score = math.sqrt((cm[0] - self.circle_center[0]) ** 2 + (cm[1] - self.circle_center[1]) ** 2)
                dt1 = self.circle_rads[0] - position_score
                dt2 = self.circle_rads[1] - position_score
                dist_normalizer = math.sqrt(object.shape[-1] ** 2 + object.shape[-2] ** 2)
                if dt2 < 0:
                    position_score = 1 - dt2 / dist_normalizer  # norm to (0, 0.5)
                elif dt1 > 0:
                    position_score = 1 - dt1 / dist_normalizer
                else:
                    position_score = 1

                shape_dist_scores = torch.tensor([des.distance(polygon, res) for des in self.fg_shape_descriptors],
                                                 device=dev)
                shape_score = 1 - shape_dist_scores.min()
                shape_score *= 0.8

                ellipset = fitEllipse(contour.astype(np.int))
                v = np.zeros((2,), dtype=np.float)
                v[1] = ((object.shape[0] / 2) - ellipset[0][0])
                v[0] = (ellipset[0][1] - (object.shape[1] / 2))
                v = v / np.clip(np.linalg.norm(v), a_min=1e-6, a_max=np.inf)
                slant = np.deg2rad(360 - ellipset[2])
                u = np.array([np.cos(slant), np.sin(slant)])
                shape_score += abs(np.dot(u, v)) * 0.2

                score = 0.5 * position_score + 0.5 * shape_score
                scores[sp_ids] += score

            # check if background objects are touching
            if not (((s_dir_edges[0][None] == bg1_sp_ids[:, None]).long().sum(0) +
                     (s_dir_edges[1][None] == bg2_sp_ids[:, None]).long().sum(0)) == 2).any():
                # penalize size variation
                diff1 = label_masses[bg1_id] - self.masses[0]
                diff2 = label_masses[bg2_id] - self.masses[1]
                if diff1 > 0 and label_masses[bg1_id] > self.masses[0] * 3 or diff1 < 0 and label_masses[bg1_id] < \
                        self.masses[0] / 3:
                    pass
                else:
                    try:
                        contour = find_contours(bg1.cpu().numpy(), level=0)
                        if len(contour) > 1:
                            scores[sp_ids] -= len(contour) / 5.0
                        contour = contour[np.array([len(cnt) for cnt in contour]).argmax()]
                        poly_chain = torch.from_numpy(approximate_polygon(contour, tolerance=0.0)).to(dev)
                        if poly_chain.shape[0] <= 3:
                            raise Exception()
                        polygon = Polygon2d(poly_chain)
                        shape_dist_scores = torch.tensor(
                            [des.distance(polygon, res * 10) for des in self.outer_cntr_ds], device=dev)
                        # shape_score = (torch.sigmoid((((1 - shape_dist_scores.min()) * 6.5).exp() / torch.tensor([6.5], device=dev).exp()) * 6 - 3) * 1.2597) - 0.2
                        shape_score = 1 - shape_dist_scores.min()
                        scores[bg1_sp_ids] += shape_score
                    except Exception as e:
                        print(e)
                if diff2 > 0 and label_masses[bg2_id] > self.masses[1] * 3 or diff2 < 0 and label_masses[bg2_id] < \
                        self.masses[1] / 3:
                    pass
                else:
                    try:
                        contour = find_contours(bg2.cpu().numpy(), level=0)
                        if len(contour) > 1:
                            scores[sp_ids] -= len(contour) / 5.0
                        contour = contour[np.array([len(cnt) for cnt in contour]).argmax()]
                        poly_chain = torch.from_numpy(approximate_polygon(contour, tolerance=0.0)).to(dev)
                        if poly_chain.shape[0] <= 3:
                            raise Exception()
                        polygon = Polygon2d(poly_chain)
                        shape_dist_scores = torch.tensor(
                            [des.distance(polygon, res * 10) for des in self.inner_cntr_ds], device=dev)
                        # shape_score = (torch.sigmoid((((1 - shape_dist_scores.min()) * 6.5).exp() / torch.tensor([6.5], device=dev).exp()) * 6 - 3) * 1.2597) - 0.2
                        shape_score = 1 - shape_dist_scores.min()
                        scores[bg2_sp_ids] += shape_score
                    except Exception as e:
                        print(e)

            scores[false_obj_sp_ids] = 0.0
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                print(Warning("NaN or inf in scores this should not happen"))
            if edge_score:
                edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]
                edge_scores = scores[edges].max(dim=0).values
                # identify wrong edges
                bg_sp = torch.arange(len(s_sp_cmrads), device=dev)[
                    (s_sp_cmrads > self.circle_rads[2]) | (s_sp_cmrads < self.circle_rads[3])]
                edge_mask = (edges[None] == bg_sp[:, None, None]).sum(0).sum(0) == 2
                edge_scores[edge_mask] = 1 - s_actions[edge_mask]
                return_scores.append(edge_scores)
            else:
                return_scores.append(scores)

        return_scores = torch.cat(return_scores)
        return return_scores


class LeptinDataReward2DTurning(RewardFunctionAbc):

    def __init__(self, device="cuda:0", *args, **kwargs):
        """
        For this data we use 4 kinds of shape descriptors. Outer contour of the whole cell constellation as well as its inner contour.
        Both will be used for the reward of the background superpixels.
        :param shape_samples:
        """
        source_file_wtsd = "/g/kreshuk/data/leptin/sourabh_data_v1/Segmentation_results_fused_tp_1_ch_0_Masked_WatershedBoundariesMergeTreeFilter_Out1.tif"
        source_file_wtsd = "/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/Masked_WatershedBoundariesMergeTreeFilter_Out1.h5"
        # wtsd = torch.from_numpy(np.array(imread(source_file_wtsd).astype(np.long))).to(device)
        wtsd = torch.from_numpy(h5py.File(source_file_wtsd, "r")["data"][:].astype(np.long)).to(device)
        slices = [0, 157, 316]
        label_1 = [1359, 886, 1240]
        label_2 = [1172, 748, 807]
        label_3 = [364, 1148, 1447]
        m1, m2, m3, m4, m5 = [], [], [], [], []
        self.outer_cntr_ds, self.inner_cntr_ds, self.celltype_1_ds, self.celltype_2_ds, self.celltype_3_ds = [], [], [], [], []
        for slc, l1, l2, l3 in zip(slices, label_1, label_2, label_3):
            bg = wtsd[:, slc, :] == 1
            bg_cnt = find_contours(bg.cpu().numpy(), level=0)
            cnt1 = bg_cnt[0] if bg_cnt[0].shape[0] > bg_cnt[1].shape[0] else bg_cnt[1]
            cnt2 = bg_cnt[1] if bg_cnt[0].shape[0] > bg_cnt[1].shape[0] else bg_cnt[0]
            for m, cnt in zip([m1, m2], [cnt1, cnt2]):
                mask = torch.zeros_like(wtsd[:, slc, :]).cpu()
                mask[np.round(cnt[:, 0]), np.round(cnt[:, 1])] = 1
                m.append(torch.from_numpy(binary_fill_holes(mask.long().cpu().numpy())).to(device).sum().item())

            mask = wtsd[:, slc, :] == l1
            m3.append(mask.long().sum().item())
            cnt3 = find_contours(mask.cpu().numpy(), level=0)[0]
            mask = wtsd[:, slc, :] == l2
            m4.append(mask.long().sum().item())
            cnt4 = find_contours(mask.cpu().numpy(), level=0)[0]
            mask = wtsd[:, slc, :] == l3
            m5.append(mask.long().sum().item())
            cnt5 = find_contours(mask.cpu().numpy(), level=0)[0]

            # img = torch.zeros_like(wtsd[:, slc, :]).cpu()
            # img[cnt1[:, 0], cnt1[:, 1]] = 1
            # plt.imshow(img);plt.show()
            # img = torch.zeros_like(wtsd[:, slc, :]).cpu()
            # img[cnt2[:, 0], cnt2[:, 1]] = 1
            # plt.imshow(img);plt.show()
            # img = torch.zeros_like(wtsd[:, slc, :]).cpu()
            # img[cnt3[:, 0], cnt3[:, 1]] = 1
            # plt.imshow(img);plt.show()
            # img = torch.zeros_like(wtsd[:, slc, :]).cpu()
            # img[cnt4[:, 0], cnt4[:, 1]] = 1
            # plt.imshow(img);plt.show()
            # img = torch.zeros_like(wtsd[:, slc, :]).cpu()
            # img[cnt5[:, 0], cnt5[:, 1]] = 1
            # plt.imshow(img);plt.show()

            self.outer_cntr_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt1, tolerance=1.2)).to(device)))
            self.inner_cntr_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt2, tolerance=1.2)).to(device)))
            self.celltype_1_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt3, tolerance=1.2)).to(device)))
            self.celltype_2_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt4, tolerance=1.2)).to(device)))
            self.celltype_3_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt5, tolerance=1.2)).to(device)))

        self.masses = [np.array(m1).mean(), np.array(m2).mean(), np.array(m3 + m4 + m5).mean()]
        self.fg_shape_descriptors = self.celltype_1_ds + self.celltype_2_ds + self.celltype_3_ds

    def __call__(self, prediction_segmentation, superpixel_segmentation, res, dir_edges, edge_score, *args, **kwargs):
        dev = prediction_segmentation.device
        return_scores = []

        for single_pred, single_sp_seg, s_dir_edges in zip(prediction_segmentation, superpixel_segmentation, dir_edges):
            scores = torch.ones(int((single_sp_seg.max()) + 1, ), device=dev) * 0.5
            if single_pred.max() == 0:  # image is empty
                return_scores.append(scores - 0.5)
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
            # get the objects that are torching the patch boarder as for them we cannot compute a relieable sim score
            false_obj_mask = label_masses < 4
            false_obj_mask[bg1_id] = False
            false_obj_mask[bg2_id] = False
            # everything else are potential objects
            potenial_obj_mask = (false_obj_mask == False)
            potenial_obj_mask[bg1_id] = False
            potenial_obj_mask[bg2_id] = False
            potential_object_ids = torch.nonzero(potenial_obj_mask).squeeze(1)  # object label IDs

            bg1 = one_hot[bg1_id]  # get background masks
            bg2 = one_hot[bg2_id]  # get background masks
            objects = one_hot[potential_object_ids]  # get object masks
            false_obj_sp_ids = torch.unique((single_sp_seg[None] + 1) * one_hot[false_obj_mask])[1:] - 1
            bg1_sp_ids = torch.unique((single_sp_seg[None] + 1) * bg1)[
                         1:] - 1  # mask out the covered superpixels (need to add 1 because the single_sp_seg start from 0)
            bg2_sp_ids = torch.unique((single_sp_seg[None] + 1) * bg2)[
                         1:] - 1  # mask out the covered superpixels (need to add 1 because the single_sp_seg start from 0)
            object_sp_ids = [torch.unique((single_sp_seg[None] + 1) * obj)[1:] - 1 for obj in objects]

            # get shape descriptors for objects and get a score by comparing to self.descriptors

            for object, obj_id, sp_ids in zip(objects, potential_object_ids, object_sp_ids):
                diff = abs(label_masses[obj_id] - self.masses[2])
                if diff > self.masses[2] / 1.5:
                    score = diff / (max(label_masses[obj_id], self.masses[2]) * 2)
                    scores[sp_ids] -= score
                    continue
                try:
                    contour = find_contours(object.cpu().numpy(), level=0)
                    if len(contour) > 1:
                        scores[sp_ids] -= 0.1
                        continue
                    contour = contour[0]
                except:
                    scores[sp_ids] -= 0.1
                    continue
                poly_chain = torch.from_numpy(approximate_polygon(contour, tolerance=0.0)).to(dev)
                if poly_chain.shape[0] <= 3:
                    scores[sp_ids] -= 0.1
                    continue
                polygon = Polygon2d(poly_chain)
                dist_scores = torch.tensor([des.distance(polygon, res) for des in self.fg_shape_descriptors],
                                           device=dev)
                # project distances for objects to similarities for superpixels
                score = (torch.sigmoid((((1 - dist_scores.min()) * 6.5).exp() / torch.tensor([6.5],
                                                                                             device=dev).exp()) * 6 - 3) * 1.2597) - 0.2
                # score = torch.exp((1 - dist_scores.min()) * 4) / torch.exp(torch.tensor([4.0], device=dev))
                scores[sp_ids] += score  # do exponential scaling

            # penalize size variation
            diff1 = abs(label_masses[bg1_id] - self.masses[0])
            diff2 = abs(label_masses[bg2_id] - self.masses[1])
            if diff1 > self.masses[0] / 2:
                score = diff1 / (max(label_masses[bg1_id], self.masses[0]) * 4)
                scores[bg1_sp_ids] -= score
            else:
                try:
                    contour = find_contours(bg1.cpu().numpy(), level=0)
                    if len(contour) > 1:
                        scores[sp_ids] -= len(contour) / 20.0
                    contour = contour[np.array([len(cnt) for cnt in contour]).argmax()]
                    poly_chain = torch.from_numpy(approximate_polygon(contour, tolerance=0.0)).to(dev)
                    if poly_chain.shape[0] <= 3:
                        raise Exception()
                    polygon = Polygon2d(poly_chain)
                    dist_scores = torch.tensor([des.distance(polygon, res * 10) for des in self.outer_cntr_ds],
                                               device=dev)
                    score = (torch.sigmoid(
                        (((1 - dist_scores.min()) * 6.5).exp() / torch.tensor([6.5],
                                                                              device=dev).exp()) * 6 - 3) * 1.2597) - 0.2
                    # score = torch.exp((1 - dist_scores.min()) * 4) / torch.exp(torch.tensor([4.0], device=dev))
                    scores[bg1_sp_ids] += score
                except Exception as e:
                    print(e)
                    scores[bg1_sp_ids] -= 0.25
            if diff2 > self.masses[1] / 2:
                score = diff2 / (max(label_masses[bg2_id], self.masses[1]) * 4)
                scores[bg2_sp_ids] -= score
            else:
                try:
                    contour = find_contours(bg2.cpu().numpy(), level=0)
                    if len(contour) > 1:
                        scores[sp_ids] -= len(contour) / 20.0
                    contour = contour[np.array([len(cnt) for cnt in contour]).argmax()]
                    poly_chain = torch.from_numpy(approximate_polygon(contour, tolerance=0.0)).to(dev)
                    if poly_chain.shape[0] <= 3:
                        raise Exception()
                    polygon = Polygon2d(poly_chain)
                    dist_scores = torch.tensor([des.distance(polygon, res * 10) for des in self.inner_cntr_ds],
                                               device=dev)
                    score = (torch.sigmoid(
                        (((1 - dist_scores.min()) * 6.5).exp() / torch.tensor([6.5],
                                                                              device=dev).exp()) * 6 - 3) * 1.2597) - 0.2
                    # score = torch.exp((1 - dist_scores.min()) * 4) / torch.exp(torch.tensor([4.0], device=dev))
                    scores[bg2_sp_ids] += score
                except Exception as e:
                    print(e)
                    scores[bg2_sp_ids] -= 0.25

            # check if background objects are touching
            if (((s_dir_edges[0][None] == bg1_sp_ids[:, None]).long().sum(0) +
                 (s_dir_edges[1][None] == bg2_sp_ids[:, None]).long().sum(0)) == 2).any():
                scores[bg1_sp_ids] -= 0.25
                scores[bg2_sp_ids] -= 0.25

            scores[false_obj_sp_ids] -= 0.5
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                print(Warning("NaN or inf in scores this should not happen"))
            if edge_scores:
                edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]
                edge_scores = scores[edges].max(dim=0).values
                return_scores.append(edge_scores)
            else:
                return_scores.append(scores)
        # return scores for each superpixel

        return torch.cat(return_scores)


class LeptinDataReward2DEllipticFit(RewardFunctionAbc):

    def __init__(self, device="cuda:0", *args, **kwargs):
        """
        For this data we use 4 kinds of shape descriptors. Outer contour of the whole cell constellation as well as its inner contour.
        Both will be used for the reward of the background superpixels.
        :param shape_samples:
        plt.imshow(torch.from_numpy(h5py.File("/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/raw.h5", "r")["data"][:].astype(np.long))[:, 144, :], cmap=plt.get_cmap('inferno'), interpolation='none');plt.style.use('dark_background');plt.axis('off');plt.savefig("/g/kreshuk/hilt/projects/data/leptin_rawex.png", bbox_inches='tight')
        plt.imshow((wtsd[:, 144, :] * (((wtsd[:, 144, :]==0) + (wtsd[:, 144, :]==1))==0)).cpu(), cmap=random_label_cmap(), interpolation='none');plt.style.use('dark_background');plt.axis('off');plt.savefig("/g/kreshuk/hilt/projects/data/leptin_segex.png", bbox_inches='tight')
        """
        # source_file_wtsd = "/g/kreshuk/data/leptin/sourabh_data_v1/Segmentation_results_fused_tp_1_ch_0_Masked_WatershedBoundariesMergeTreeFilter_Out1.tif"
        source_file_wtsd = "/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/Masked_WatershedBoundariesMergeTreeFilter_Out1.h5"
        # wtsd = torch.from_numpy(np.array(imread(source_file_wtsd).astype(np.long))).to(device)
        wtsd = torch.from_numpy(h5py.File(source_file_wtsd, "r")["data"][:].astype(np.long)).to(device)
        slices = [0, 157, 316]
        slices_labels = [[1359, 1172, 364, 145, 282, 1172, 1359, 189, 809, 737],
                         [886, 748, 1148, 1422, 696, 684, 817, 854, 158, 774],
                         [1240, 807, 1447, 69, 1358, 1240, 129, 252, 62, 807]]
        m1, m2 = [], []
        # widths, heights = [], []
        self.outer_cntr_ds, self.inner_cntr_ds = [], []
        for slc, labels in zip(slices, slices_labels):
            bg = wtsd[:, slc, :] == 1
            bg_cnt = find_contours(bg.cpu().numpy(), level=0)
            cnt1 = bg_cnt[0] if bg_cnt[0].shape[0] > bg_cnt[1].shape[0] else bg_cnt[1]
            cnt2 = bg_cnt[1] if bg_cnt[0].shape[0] > bg_cnt[1].shape[0] else bg_cnt[0]
            for m, cnt in zip([m1, m2], [cnt1, cnt2]):
                mask = torch.zeros_like(wtsd[:, slc, :]).cpu()
                mask[np.round(cnt[:, 0]), np.round(cnt[:, 1])] = 1
                m.append(torch.from_numpy(binary_fill_holes(mask.long().cpu().numpy())).to(device).sum().item())
            self.outer_cntr_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt1, tolerance=1.2)).to(device)))
            self.inner_cntr_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt2, tolerance=1.2)).to(device)))
        #
        #     for l in labels:
        #         mask = wtsd[:, slc, :] == l
        #         cnt = find_contours(mask.cpu().numpy(), level=0)[0]
        #
        #         # img = torch.zeros_like(wtsd[:, slc, :]).cpu()
        #         # img[cnt[:, 0], cnt[:, 1]] = 1
        #         # plt.imshow(img);plt.show()
        #
        #         ellipseT = fitEllipse(cnt.astype(np.int))
        #         widths.append(ellipseT[1][1])
        #         heights.append(ellipseT[1][0])
        #
        #
        #
        # self.masses = [np.array(m1).mean(), np.array(m2).mean()]
        # self.expected_ratio = np.array(widths).mean() / np.array(heights).mean()
        self.expected_ratio = 5.573091
        self.masses = [290229.3, 97252.3]

    def __call__(self, prediction_segmentation, superpixel_segmentation, res, dir_edges, edge_score, *args, **kwargs):
        dev = prediction_segmentation.device
        return_scores = []

        for single_pred, single_sp_seg, s_dir_edges in zip(prediction_segmentation, superpixel_segmentation, dir_edges):
            scores = torch.ones(int((single_sp_seg.max()) + 1, ), device=dev) * 0.5
            if single_pred.max() == 0:  # image is empty
                return_scores.append(scores - 0.5)
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
            # get the objects that are torching the patch boarder as for them we cannot compute a relieable sim score
            false_obj_mask = label_masses < 4
            false_obj_mask[bg1_id] = False
            false_obj_mask[bg2_id] = False
            # everything else are potential objects
            potenial_obj_mask = (false_obj_mask == False)
            potenial_obj_mask[bg1_id] = False
            potenial_obj_mask[bg2_id] = False
            potential_object_ids = torch.nonzero(potenial_obj_mask).squeeze(1)  # object label IDs

            bg1 = one_hot[bg1_id]  # get background masks
            bg2 = one_hot[bg2_id]  # get background masks
            objects = one_hot[potential_object_ids]  # get object masks
            false_obj_sp_ids = torch.unique((single_sp_seg[None] + 1) * one_hot[false_obj_mask])[1:] - 1
            bg1_sp_ids = torch.unique((single_sp_seg[None] + 1) * bg1)[
                         1:] - 1  # mask out the covered superpixels (need to add 1 because the single_sp_seg start from 0)
            bg2_sp_ids = torch.unique((single_sp_seg[None] + 1) * bg2)[
                         1:] - 1  # mask out the covered superpixels (need to add 1 because the single_sp_seg start from 0)
            object_sp_ids = [torch.unique((single_sp_seg[None] + 1) * obj)[1:] - 1 for obj in objects]

            # get shape descriptors for objects and get a score by comparing to self.descriptors

            for object, obj_id, sp_ids in zip(objects, potential_object_ids, object_sp_ids):
                try:
                    contour = find_contours(object.cpu().numpy(), level=0)
                    if len(contour) > 1:
                        scores[sp_ids] -= 0.1
                        continue
                    contour = contour[0]
                except:
                    scores[sp_ids] -= 0.1
                    continue
                poly_chain = torch.from_numpy(approximate_polygon(contour, tolerance=0.0)).to(dev)
                if poly_chain.shape[0] <= 3:
                    scores[sp_ids] -= 0.1
                    continue
                score = 0

                try:
                    ellipseT = fitEllipse(contour.astype(np.int))
                    ratio = ellipseT[1][1] / (ellipseT[1][0] + 1e-10)
                    ratio_diff = abs(self.expected_ratio - ratio)

                    v = np.zeros((2,), dtype=np.float)
                    v[1] = ((object.shape[0] / 2) - ellipseT[0][0])
                    v[0] = (ellipseT[0][1] - (object.shape[1] / 2))
                    v = v / np.clip(np.linalg.norm(v), a_min=1e-6, a_max=np.inf)
                    slant = np.deg2rad(360 - ellipseT[2])
                    u = np.array([np.cos(slant), np.sin(slant)])
                    score += abs(np.dot(u, v)) / 4

                    if ratio_diff > self.expected_ratio / 2:
                        score += 0.1
                    elif ratio_diff > self.expected_ratio / 3:
                        score += 0.2
                    elif ratio_diff > self.expected_ratio / 4:
                        score += 0.3
                    elif ratio_diff > self.expected_ratio / 6:
                        score += 0.4
                    else:
                        score += 0.5
                except Exception as e:
                    print(e)
                    score -= 0.1

                # ellipse = ((ellipseT[0][1], ellipseT[0][0]), (ellipseT[1][1], ellipseT[1][0]), 360-ellipseT[2])
                # plt.imshow(object.cpu()[..., None] * 200 + cv2.ellipse(np.zeros(shape=[749, 692, 3], dtype=np.uint8), ellipse, (23, 184, 80), 3))
                # plt.show()

                scores[sp_ids] += score  # do exponential scaling

            # penalize size variation
            diff1 = abs(label_masses[bg1_id] - self.masses[0])
            diff2 = abs(label_masses[bg2_id] - self.masses[1])
            if diff1 > self.masses[0] / 2:
                score = diff1 / (max(label_masses[bg1_id], self.masses[0]) * 4)
                scores[bg1_sp_ids] -= score
            else:
                try:
                    contour = find_contours(bg1.cpu().numpy(), level=0)
                    if len(contour) > 1:
                        scores[sp_ids] -= len(contour) / 20.0
                    contour = contour[np.array([len(cnt) for cnt in contour]).argmax()]
                    poly_chain = torch.from_numpy(approximate_polygon(contour, tolerance=0.0)).to(dev)
                    if poly_chain.shape[0] <= 3:
                        raise Exception()
                    polygon = Polygon2d(poly_chain)
                    dist_scores = torch.tensor([des.distance(polygon, res * 10) for des in self.outer_cntr_ds],
                                               device=dev)
                    score = (torch.sigmoid(
                        (((1 - dist_scores.min()) * 6.5).exp() / torch.tensor([6.5],
                                                                              device=dev).exp()) * 6 - 3) * 1.2597) - 0.2
                    # score = torch.exp((1 - dist_scores.min()) * 4) / torch.exp(torch.tensor([4.0], device=dev))
                    scores[bg1_sp_ids] += score
                except Exception as e:
                    print(e)
                    scores[bg1_sp_ids] -= 0.25
            if diff2 > self.masses[1] / 2:
                score = diff2 / (max(label_masses[bg2_id], self.masses[1]) * 4)
                scores[bg2_sp_ids] -= score
            else:
                try:
                    contour = find_contours(bg2.cpu().numpy(), level=0)
                    if len(contour) > 1:
                        scores[sp_ids] -= len(contour) / 20.0
                    contour = contour[np.array([len(cnt) for cnt in contour]).argmax()]
                    poly_chain = torch.from_numpy(approximate_polygon(contour, tolerance=0.0)).to(dev)
                    if poly_chain.shape[0] <= 3:
                        raise Exception()
                    polygon = Polygon2d(poly_chain)
                    dist_scores = torch.tensor([des.distance(polygon, res * 10) for des in self.inner_cntr_ds],
                                               device=dev)
                    score = (torch.sigmoid(
                        (((1 - dist_scores.min()) * 6.5).exp() / torch.tensor([6.5],
                                                                              device=dev).exp()) * 6 - 3) * 1.2597) - 0.2
                    # score = torch.exp((1 - dist_scores.min()) * 4) / torch.exp(torch.tensor([4.0], device=dev))
                    scores[bg2_sp_ids] += score
                except Exception as e:
                    print(e)
                    scores[bg2_sp_ids] -= 0.25

            # check if background objects are touching
            if (((s_dir_edges[0][None] == bg1_sp_ids[:, None]).long().sum(0) +
                 (s_dir_edges[1][None] == bg2_sp_ids[:, None]).long().sum(0)) == 2).any():
                scores[bg1_sp_ids] -= 0.25
                scores[bg2_sp_ids] -= 0.25

            scores[false_obj_sp_ids] -= 0.5
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                print(Warning("NaN or inf in scores this should not happen"))
            if edge_scores:
                edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]
                edge_scores = scores[edges].max(dim=0).values
                return_scores.append(edge_scores)
            else:
                return_scores.append(scores)
        # return scores for each superpixel

        return torch.cat(return_scores)


if __name__ == "__main__":
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from utils.general import multicut_from_probas, calculate_gt_edge_costs
    from glob import glob
    import os
    from threading import Thread


    label_cm = random_label_cmap(zeroth=1.0)
    label_cm.set_bad(alpha=0)
    edge_cmap = cm.get_cmap(name="cool")
    edge_cmap.set_bad(alpha=0)
    dev = "cuda:0"
    # get a few images and extract some gt objects used ase shape descriptors that we want to compare against
    dir = "/g/kreshuk/kaziakhm/leptin_data/processed/v4_dwtrsd/val"
    fnames_pix = sorted(glob(os.path.join(dir, 'pix_data/*.h5')))
    fnames_graph = sorted(glob(os.path.join(dir, 'graph_data/*.h5')))

    id_set1 = [97, 99, 100, 104, 105, 111, 110, 113]
    bg_ids = [0, 31]


    def merge_sp_ids(seg, ids):
        lbl = min(ids)
        for id in ids:
            seg[seg == id] = lbl

        mask = seg[None] == torch.unique(seg)[:, None, None]
        seg = (mask * (torch.arange(len(torch.unique(seg)), device=seg.device)[:, None, None] + 1)).sum(0) - 1
        return seg

    # fnames_graph = ['/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/true_val/graph_data/graph_125.h5']
    # fnames_pix= ['/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/true_val/pix_data/pix_125.h5']

    def workerfunc(pfname, gfname):
        p_file = h5py.File(pfname, 'r')
        g_file = h5py.File(gfname, 'r')
        raw = p_file['raw'][:].astype(np.float)
        suppix = torch.from_numpy(p_file['node_labeling'][:]).to(dev).float()
        onehot = suppix[None] == torch.unique(suppix)[:, None, None]
        masses = onehot.flatten(1).sum(1)
        meshgrid = torch.stack(torch.meshgrid(torch.arange(suppix.shape[0], device=dev),
                                              torch.arange(suppix.shape[1], device=dev)))
        cms = (onehot[:, None] * meshgrid[None]).flatten(2).sum(2) / masses[:, None]
        circle_center = cms[masses < 1000].mean(0).round().long().cpu().numpy()
        raw /= raw.max()
        raw = torch.from_numpy(raw)
        # circle_center = np.array([390, 340])
        circle_rads = [210, 270, 100, 340]
        exact_circle_diameter = [345, 353, 603, 611]
        show_circle(raw, circle_center, circle_rads)

    # slices = []
    # chunksize = 10
    # n = len(fnames_pix)
    # for i in range(int(n//chunksize)):
    #     slices.append(slice(i*chunksize, (i+1)*chunksize))
    # slices.append(slice(int((n//chunksize) * chunksize), n))
    # # for slice in slices:
    # #     workers = []
    # for pfname, gfname in zip(fnames_pix, fnames_graph):
    #     workerfunc(pfname, gfname)
    #     # workers.append(Thread(target=workerfunc, args=(pfname, )))
    #     # workers[-1].start()
    #     # workers[-1].join()
    #     # for worker in workers:
    #     #     worker.join()

    for i in range(1):
        g_file = h5py.File(fnames_graph[i], 'r')
        pix_file = h5py.File(fnames_pix[i], 'r')
        superpixel_seg = pix_file['node_labeling'][:]
        gt_seg = pix_file['gt'][:]
        # gt_seg = torch.zeros_like(superpixel_seg)
        superpixel_seg = torch.from_numpy(superpixel_seg.astype(np.int64)).to(dev)
        gt_seg = torch.from_numpy(gt_seg.astype(np.int64)).to(dev)

        all_ids = torch.unique(gt_seg)
        all_ids = all_ids[all_ids != bg_ids[0]]
        all_ids = all_ids[all_ids != bg_ids[1]]

        probas = g_file['edge_feat'][:, 0]  # take initial edge features as weights

        # make sure probas are probas and get a sample prediction
        probas -= probas.min()
        probas /= (probas.max() + 1e-6)
        pred_seg = multicut_from_probas(superpixel_seg.cpu().numpy(), g_file['edges'][:].T, probas)
        pred_seg = torch.from_numpy(pred_seg.astype(np.int64)).to(dev)

        # relabel to consecutive integers:
        mask =gt_seg[None] == torch.unique(gt_seg)[:, None, None]
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

        noisy_gt_seg = merge_sp_ids(gt_seg.squeeze().clone(), all_ids[:int(all_ids.shape[0] / 2)].tolist())
        gt_edges = calculate_gt_edge_costs(edges.T, superpixel_seg.squeeze(), noisy_gt_seg.squeeze(), thresh=0.5)

        mc_seg_old = None
        for itr in range(100):
            # actions = gt_edges + torch.randn_like(gt_edges) * (itr / 100)
            # actions -= actions.min()
            # actions /= actions.max()
            actions = gt_edges
            # actions = torch.zeros_like(gt_edges)
            # actions[:20] = 1.0

            mc_seg = multicut_from_probas(superpixel_seg.cpu().numpy(), edges.T.cpu().numpy(), actions)
            mc_seg = torch.from_numpy(mc_seg.astype(np.int64)).to(dev)

            # relabel to consecutive integers:
            mask = mc_seg[None] == torch.unique(mc_seg)[:, None, None]
            mc_seg = (mask * (torch.arange(len(torch.unique(mc_seg)), device=dev)[:, None, None] + 1)).sum(0) - 1
            # add batch dimension
            pred_seg = pred_seg[None]
            noisy_gt_seg = noisy_gt_seg[None]
            mc_seg = mc_seg[None]

            f = LeptinDataRotatedRectRewards()
            # f.get_gaussians(.02, 0.13, .12, 0.2, 0.4, .3)
            # plt.imshow(pix_file['raw'][:]);plt.show()
            # rewards2 = f(gt_seg.long(), superpixel_seg.long(), dir_edges=[dir_edges], res=100)
            edge_angles, sp_feat, sp_rads, cms, masses = get_angles_smass_in_rag(edges, superpixel_seg.long())
            edge_rewards = f(mc_seg.long(), superpixel_seg[None].long(), dir_edges=[dir_edges], res=50, edge_score=True,
                             sp_cmrads=[sp_rads], actions=[actions], sp_cms=[cms], sp_masses=[masses])

            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            frame_rew, scores_rew, bnd_mask = get_colored_edges_in_sseg(superpixel_seg[None].float(), edges, edge_rewards)
            frame_act, scores_act, _ = get_colored_edges_in_sseg(superpixel_seg[None].float(), edges, 1 - actions.squeeze())

            bnd_mask = torch.from_numpy(dilation(bnd_mask.cpu().numpy()))

            mc_seg = sync_segmentations(mc_seg_old, mc_seg) if mc_seg_old is not None else mc_seg
            mc_seg_old = mc_seg.clone()
            masked_mc_seg = mc_seg.squeeze().float().clone()
            masked_mc_seg[bnd_mask] = np.nan

            frame_rew = np.stack([dilation(frame_rew.cpu().numpy()[..., i]) for i in range(3)], -1)
            frame_act = np.stack([dilation(frame_act.cpu().numpy()[..., i]) for i in range(3)], -1)
            scores_rew = np.stack([dilation(scores_rew.cpu().numpy()[..., i]) for i in range(3)], -1)
            scores_act = np.stack([dilation(scores_act.cpu().numpy()[..., i]) for i in range(3)], -1)

            ax[0, 0].imshow(frame_rew, interpolation="none")
            ax[0, 0].imshow(masked_mc_seg.cpu(), cmap=label_cm, alpha=0.8, interpolation="none")
            ax[0, 0].set_title("rewards 1:g, 0:r")
            ax[0, 0].axis('off')
            ax[0, 1].imshow(frame_act, interpolation="none")
            ax[0, 1].imshow(masked_mc_seg.cpu(), cmap=label_cm, alpha=0.8, interpolation="none")
            ax[0, 1].set_title("actions 0:g, 1:r")
            ax[0, 1].axis('off')

            ax[1, 0].imshow(mc_seg.cpu().squeeze(), cmap=random_label_cmap(), interpolation="none")
            ax[1, 0].set_title("pred")
            ax[1, 0].axis('off')
            ax[1, 1].imshow(gt_seg.cpu().squeeze(), cmap=random_label_cmap(), interpolation="none")
            ax[1, 1].set_title("gt")
            ax[1, 1].axis('off')


            # plt.savefig("/g/kreshuk/hilt/projects/RLForSeg/data/test.png", bbox_inches='tight')
            plt.show()
            a = 1

        # rewards1 = f(pred_seg.long(), superpixel_seg.long(), dir_edges=[dir_edges], res=100)
