import matplotlib
# matplotlib.use('TkAgg')
from rewards.reward_abc import RewardFunctionAbc
from skimage.measure import approximate_polygon,  find_contours
from skimage.draw import polygon_perimeter
from utils.polygon_2d import Polygon2d
import torch
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


    def __call__(self, prediction_segmentation, superpixel_segmentation, res, dir_edges, *args, **kwargs):
        dev = prediction_segmentation.device
        return_scores = []

        for single_pred, single_sp_seg, s_dir_edges in zip(prediction_segmentation, superpixel_segmentation, dir_edges):
            scores = torch.ones(int((single_sp_seg.max()) + 1,), device=dev) * 0.5
            if single_pred.max() == 0:  # image is empty
                return_scores.append(scores - 0.5)
                continue
            # get one-hot representation
            one_hot = torch.zeros((int(single_pred.max()) + 1, ) + single_pred.size(), device=dev, dtype=torch.long) \
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
            bg1_sp_ids = torch.unique((single_sp_seg[None] + 1) * bg1)[1:] - 1  # mask out the covered superpixels (need to add 1 because the single_sp_seg start from 0)
            bg2_sp_ids = torch.unique((single_sp_seg[None] + 1) * bg2)[1:] - 1  # mask out the covered superpixels (need to add 1 because the single_sp_seg start from 0)
            object_sp_ids = [torch.unique((single_sp_seg[None] + 1) * obj)[1:] - 1 for obj in objects]

            #get shape descriptors for objects and get a score by comparing to self.descriptors

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
                dist_scores = torch.tensor([des.distance(polygon, res) for des in self.fg_shape_descriptors], device=dev)
                #project distances for objects to similarities for superpixels
                score = (torch.sigmoid((((1 - dist_scores.min()) * 6.5).exp() / torch.tensor([6.5], device=dev).exp()) * 6 - 3) * 1.2597) - 0.2
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
                    dist_scores = torch.tensor([des.distance(polygon, res*10) for des in self.outer_cntr_ds], device=dev)
                    score = (torch.sigmoid(
                        (((1 - dist_scores.min()) * 6.5).exp() / torch.tensor([6.5], device=dev).exp()) * 6 - 3) * 1.2597) - 0.2
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
                    dist_scores = torch.tensor([des.distance(polygon, res*10) for des in self.inner_cntr_ds], device=dev)
                    score = (torch.sigmoid(
                        (((1 - dist_scores.min()) * 6.5).exp() / torch.tensor([6.5], device=dev).exp()) * 6 - 3) * 1.2597) - 0.2
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
            return_scores.append(scores)
        #return scores for each superpixel

        return torch.cat(return_scores)


class LeptinDataReward2DEllipticFit(RewardFunctionAbc):

    def __init__(self, device="cuda:0", *args, **kwargs):
        """
        For this data we use 4 kinds of shape descriptors. Outer contour of the whole cell constellation as well as its inner contour.
        Both will be used for the reward of the background superpixels.
        :param shape_samples:
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


    def __call__(self, prediction_segmentation, superpixel_segmentation, res, dir_edges, *args, **kwargs):
        dev = prediction_segmentation.device
        return_scores = []

        for single_pred, single_sp_seg, s_dir_edges in zip(prediction_segmentation, superpixel_segmentation, dir_edges):
            scores = torch.ones(int((single_sp_seg.max()) + 1,), device=dev) * 0.5
            if single_pred.max() == 0:  # image is empty
                return_scores.append(scores - 0.5)
                continue
            # get one-hot representation
            one_hot = torch.zeros((int(single_pred.max()) + 1, ) + single_pred.size(), device=dev, dtype=torch.long) \
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
            bg1_sp_ids = torch.unique((single_sp_seg[None] + 1) * bg1)[1:] - 1  # mask out the covered superpixels (need to add 1 because the single_sp_seg start from 0)
            bg2_sp_ids = torch.unique((single_sp_seg[None] + 1) * bg2)[1:] - 1  # mask out the covered superpixels (need to add 1 because the single_sp_seg start from 0)
            object_sp_ids = [torch.unique((single_sp_seg[None] + 1) * obj)[1:] - 1 for obj in objects]

            #get shape descriptors for objects and get a score by comparing to self.descriptors

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
                    dist_scores = torch.tensor([des.distance(polygon, res*10) for des in self.outer_cntr_ds], device=dev)
                    score = (torch.sigmoid(
                        (((1 - dist_scores.min()) * 6.5).exp() / torch.tensor([6.5], device=dev).exp()) * 6 - 3) * 1.2597) - 0.2
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
                    dist_scores = torch.tensor([des.distance(polygon, res*10) for des in self.inner_cntr_ds], device=dev)
                    score = (torch.sigmoid(
                        (((1 - dist_scores.min()) * 6.5).exp() / torch.tensor([6.5], device=dev).exp()) * 6 - 3) * 1.2597) - 0.2
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
            return_scores.append(scores)
        #return scores for each superpixel

        return torch.cat(return_scores)


if __name__ == "__main__":
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    from utils.general import multicut_from_probas, calculate_gt_edge_costs
    from glob import glob
    import os

    dev = "cuda:0"
    # get a few images and extract some gt objects used ase shape descriptors that we want to compare against
    dir = "/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/train"
    fnames_pix = sorted(glob(os.path.join(dir, 'pix_data/*.h5')))
    fnames_graph = sorted(glob(os.path.join(dir, 'graph_data/*.h5')))


    g_file = h5py.File(fnames_graph[42], 'r')
    pix_file = h5py.File(fnames_pix[42], 'r')
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
    superpixel_seg = (mask * (torch.arange(len(torch.unique(superpixel_seg)), device=dev)[:, None, None] + 1)).sum(0) - 1
    mask = pred_seg[None] == torch.unique(pred_seg)[:, None, None]
    pred_seg = (mask * (torch.arange(len(torch.unique(pred_seg)), device=dev)[:, None, None] + 1)).sum(0) - 1
    # assert the segmentations are consecutive integers
    assert pred_seg.max() == len(torch.unique(pred_seg)) - 1
    assert superpixel_seg.max() == len(torch.unique(superpixel_seg)) - 1


    edges = torch.from_numpy(compute_rag(superpixel_seg.cpu().numpy()).uvIds().astype(np.long)).T.to(dev)
    dir_edges = torch.stack((torch.cat((edges[0], edges[1])), torch.cat((edges[1], edges[0]))))

    gt_edges = calculate_gt_edge_costs(edges.T.cpu().numpy(), superpixel_seg.squeeze().cpu().numpy(), gt_seg.squeeze().cpu().numpy())
    mc_gt_seg = multicut_from_probas(superpixel_seg.cpu().numpy(), edges.T.cpu().numpy(), gt_edges)
    mc_gt_seg = torch.from_numpy(mc_gt_seg.astype(np.int64)).to(dev)

    # relabel to consecutive integers:
    mask = mc_gt_seg[None] == torch.unique(mc_gt_seg)[:, None, None]
    mc_gt_seg = (mask * (torch.arange(len(torch.unique(mc_gt_seg)), device=dev)[:, None, None] + 1)).sum(0) - 1
    # add batch dimension
    pred_seg = pred_seg[None]
    superpixel_seg = superpixel_seg[None]
    gt_seg = gt_seg[None]
    mc_gt_seg = mc_gt_seg[None]

    f = LeptinDataReward2DEllipticFit()
    # rewards2 = f(gt_seg.long(), superpixel_seg.long(), dir_edges=[dir_edges], res=100)
    rewards2 = f(mc_gt_seg.long(), superpixel_seg.long(), dir_edges=[dir_edges], res=100)

    # rewards1 = f(pred_seg.long(), superpixel_seg.long(), dir_edges=[dir_edges], res=100)
    a=1
