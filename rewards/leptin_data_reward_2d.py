from rewards.reward_abc import RewardFunctionAbc
from skimage.measure import approximate_polygon,  find_contours
from skimage.draw import polygon_perimeter
from utils.polygon_2d import Polygon2d
import torch
from elf.segmentation.features import compute_rag
from tifffile import imread
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

class LeptinDataReward2D(RewardFunctionAbc):

    def __init__(self, device="cuda:0", *args, **kwargs):
        """
        For this data we use 4 kinds of shape descriptors. Outer contour of the whole cell constellation as well as its inner contour.
        Both will be used for the reward of the background superpixels.
        :param shape_samples:
        """
        source_file_wtsd = "/g/kreshuk/data/leptin/sourabh_data_v1/Segmentation_results_fused_tp_1_ch_0_Masked_WatershedBoundariesMergeTreeFilter_Out1.tif"
        wtsd = torch.from_numpy(np.array(imread(source_file_wtsd).astype(np.long))).to(device)
        slices = [0, 157, 316]
        label_1 = [1359, 886, 1240]
        label_2 = [1172, 748, 807]
        label_3 = [364, 1148, 1447]
        self.outer_cntr_ds, self.inner_cntr_ds, self.celltype_1_ds, self.celltype_2_ds, self.celltype_3_ds = [], [], [], [], []
        for slc, l1, l2, l3 in zip(slices, label_1, label_2, label_3):
            bg = wtsd[:, slc, :] == 1
            bg_cnt = find_contours(bg.cpu().numpy(), level=0)
            cnt1 = bg_cnt[0] if bg_cnt[0].shape[0] > bg_cnt[1].shape[0] else bg_cnt[1]
            cnt2 = bg_cnt[1] if bg_cnt[0].shape[0] > bg_cnt[1].shape[0] else bg_cnt[0]
            cnt3 = find_contours((wtsd[:, slc, :] == l1).cpu().numpy(), level=0)[0]
            cnt4 = find_contours((wtsd[:, slc, :] == l2).cpu().numpy(), level=0)[0]
            cnt5 = find_contours((wtsd[:, slc, :] == l3).cpu().numpy(), level=0)[0]

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

            self.outer_cntr_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt1, tolerance=1.2)).to(dev)))
            self.inner_cntr_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt2, tolerance=1.2)).to(dev)))
            self.celltype_1_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt3, tolerance=1.2)).to(dev)))
            self.celltype_2_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt4, tolerance=1.2)).to(dev)))
            self.celltype_3_ds.append(Polygon2d(torch.from_numpy(approximate_polygon(cnt5, tolerance=1.2)).to(dev)))

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
            bg1_mask[bg1_id] = True
            bg2_mask = torch.zeros_like(label_masses).bool()
            bg2_id = label_masses[bg1_mask == False].argmax()
            bg2_mask[bg2_id] = True
            # get the objects that are torching the patch boarder as for them we cannot compute a relieable sim score
            false_obj_mask = (label_masses < 15) | (label_masses > 50**2)
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

            for object, sp_ids in zip(objects, object_sp_ids):
                try:
                    contour = find_contours(object.cpu().numpy(), level=0)[0]
                except:
                    a=1
                poly_chain = torch.from_numpy(approximate_polygon(contour, tolerance=0.0)).to(dev)
                if poly_chain.shape[0] <= 3:
                    scores[sp_ids] -= 0.1
                    continue
                polygon = Polygon2d(poly_chain)
                dist_scores = torch.tensor([des.distance(polygon, res) for des in self.fg_shape_descriptors], device=dev)
                #project distances for objects to similarities for superpixels
                scores[sp_ids] += torch.exp((1 - dist_scores.min()) * 8) / torch.exp(torch.tensor([8.0], device=dev))  # do exponential scaling
                if torch.isnan(scores).any() or torch.isinf(scores).any():
                    a=1

            scores[false_obj_sp_ids] -= 0.5
            # check if background objects are touching
            if (((s_dir_edges[0][None] == bg1_sp_ids[:, None]).long().sum(0) +
                 (s_dir_edges[1][None] == bg2_sp_ids[:, None]).long().sum(0)) == 2).any():
                scores[bg1_sp_ids] -= 0.5
                scores[bg2_sp_ids] -= 0.5
            dist_scores = torch.tensor([des.distance(polygon, res) for des in self.outer_cntr_ds], device=dev)
            scores[bg1_sp_ids] += torch.exp((1 - dist_scores.min()) * 8) / torch.exp(torch.tensor([8.0], device=dev))
            dist_scores = torch.tensor([des.distance(polygon, res) for des in self.inner_cntr_ds], device=dev)
            scores[bg2_sp_ids] += torch.exp((1 - dist_scores.min()) * 8) / torch.exp(torch.tensor([8.0], device=dev))

            return_scores.append(scores)
            #return scores for each superpixel

        return torch.cat(return_scores)


if __name__ == "__main__":
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    from utils.general import multicut_from_probas
    from glob import glob
    import os

    dev = "cuda:0"
    # get a few images and extract some gt objects used ase shape descriptors that we want to compare against
    dir = "/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/train"
    fnames_pix = sorted(glob(os.path.join(dir, 'pix_data/*.h5')))
    fnames_graph = sorted(glob(os.path.join(dir, 'graph_data/*.h5')))


    g_file = h5py.File(fnames_graph[42], 'r')
    superpixel_seg = g_file['node_labeling'][:]
    superpixel_seg = torch.from_numpy(superpixel_seg.astype(np.int64)).to(dev)

    probas = g_file['edge_feat'][:, 0]  # take initial edge features as weights

    # make sure probas are probas and get a sample prediction
    probas -= probas.min()
    probas /= (probas.max() + 1e-6)
    pred_seg = multicut_from_probas(superpixel_seg.cpu().numpy(), g_file['edges'][:].T, probas)
    pred_seg = torch.from_numpy(pred_seg.astype(np.int64)).to(dev)

    # relabel to consecutive integers:
    mask = superpixel_seg[None] == torch.unique(superpixel_seg)[:, None, None]
    superpixel_seg = (mask * (torch.arange(len(torch.unique(superpixel_seg)), device=dev)[:, None, None] + 1)).sum(0) - 1
    mask = pred_seg[None] == torch.unique(pred_seg)[:, None, None]
    pred_seg = (mask * (torch.arange(len(torch.unique(pred_seg)), device=dev)[:, None, None] + 1)).sum(0) - 1
    # assert the segmentations are consecutive integers
    assert pred_seg.max() == len(torch.unique(pred_seg)) - 1
    assert superpixel_seg.max() == len(torch.unique(superpixel_seg)) - 1

    edges = torch.from_numpy(compute_rag(superpixel_seg.cpu().numpy()).uvIds().astype(np.long)).T.to(dev)
    dir_edges = torch.stack((torch.cat((edges[0], edges[1])), torch.cat((edges[1], edges[0]))))

    # add batch dimension
    pred_seg = pred_seg[None]
    superpixel_seg = superpixel_seg[None]

    f = LeptinDataReward2D()
    rewards = f(pred_seg.long(), superpixel_seg.long(), dir_edges=[dir_edges], res=100)