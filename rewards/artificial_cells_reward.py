from rewards.reward_abc import RewardFunctionAbc
from skimage.measure import approximate_polygon,  find_contours
from skimage.draw import polygon_perimeter
from utils.polygon_2d import Polygon2d
import torch
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

class ArtificialCellsReward(RewardFunctionAbc):

    def __init__(self, shape_samples):
        #TODO get the descriptors for the shape samples
        dev = shape_samples.device
        self.gt_descriptors = []
        for shape_sample in shape_samples:
            shape_sample = shape_sample.cpu().numpy()
            contour = find_contours(shape_sample, level=0)[0]
            poly_approx = torch.from_numpy(approximate_polygon(contour, tolerance=1.2)).to(dev)

            self.gt_descriptors.append(Polygon2d(poly_approx))

    def __call__(self, prediction_segmentation, superpixel_segmentation, res, *args, **kwargs):
        dev = prediction_segmentation.device
        return_scores = []
        inner_halo_mask = torch.zeros(superpixel_segmentation.shape[1:], device=dev)
        inner_halo_mask[0, :] = 1
        inner_halo_mask[:, 0] = 1
        inner_halo_mask[-1, :] = 1
        inner_halo_mask[:, -1] = 1
        inner_halo_mask = inner_halo_mask.unsqueeze(0)

        for single_pred, single_sp_seg in zip(prediction_segmentation, superpixel_segmentation):
            scores = torch.ones(int((single_sp_seg.max()) + 1,), device=dev) * 0.5
            if single_pred.max() == 0:  # image is empty
                return_scores.append(scores - 0.5)
                continue
            # get one-hot representation
            one_hot = torch.zeros((int(single_pred.max()) + 1, ) + single_pred.size(), device=dev, dtype=torch.long) \
                .scatter_(0, single_pred[None], 1)

            # need masses to determine what objects can be considered background
            label_masses = one_hot.flatten(1).sum(-1)
            bg_mask = torch.zeros_like(label_masses).bool()
            bg_id = label_masses.argmax()
            bg_mask[bg_id] = True
            # get the objects that are torching the patch boarder as for them we cannot compute a relieable sim score
            invalid_obj_mask = ((one_hot * inner_halo_mask).flatten(1).sum(-1) >= 2) & (bg_mask == False)
            false_obj_mask = (label_masses < 15) | (label_masses > 50**2)
            false_obj_mask[bg_id] = False
            # everything else are potential objects
            potenial_obj_mask = (false_obj_mask == False) & (invalid_obj_mask == False)
            potenial_obj_mask[bg_id] = False
            potential_object_ids = torch.nonzero(potenial_obj_mask).squeeze(1)  # object label IDs

            bg = one_hot[bg_id]  # get background masks
            objects = one_hot[potential_object_ids]  # get object masks
            false_obj_sp_ids = torch.unique((single_sp_seg[None] + 1) * one_hot[false_obj_mask])[1:] - 1
            bg_sp_ids = [torch.unique((single_sp_seg[None] + 1) * bg_obj)[1:] - 1 for bg_obj in bg]  # mask out the covered superpixels (need to add 1 because the single_sp_seg start from 0)
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
                dist_scores = torch.tensor([des.distance(polygon, res) for des in self.gt_descriptors], device=dev)
                #project distances for objects to similarities for superpixels
                scores[sp_ids] += 1 - dist_scores.min()
                # scores[sp_ids] += torch.exp((1 - dist_scores.min()) * 8) / torch.exp(torch.tensor([8.0], device=dev))  # do exponential scaling
                if torch.isnan(scores).any() or torch.isinf(scores).any():
                    a=1

            scores[false_obj_sp_ids] -= 0.5
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

            return_scores.append(scores)
            #return scores for each superpixel

        return torch.cat(return_scores)


if __name__ == "__main__":
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    from utils.general import multicut_from_probas
    from glob import glob

    dev = "cuda:0"
    # get a few images and extract some gt objects used ase shape descriptors that we want to compare against
    dir = "/g/kreshuk/hilt/projects/data/artificial_cells"
    fnames_pix = sorted(glob('/g/kreshuk/hilt/projects/data/artificial_cells/pix_data/*.h5'))
    fnames_graph = sorted(glob('/g/kreshuk/hilt/projects/data/artificial_cells/graph_data/*.h5'))
    gt = torch.from_numpy(h5py.File(fnames_pix[42], 'r')['gt'][:]).to(dev)

    # set gt to integer labels
    _gt = torch.zeros_like(gt).long()
    for _lbl, lbl in enumerate(torch.unique(gt)):
        _gt += (gt == lbl).long() * _lbl
    gt = _gt
    sample_shapes = torch.zeros((int(gt.max()) + 1,) + gt.size(), device=dev).scatter_(0, gt[None], 1)[1:]  # 0 should be background

    g_file = h5py.File(fnames_graph[42], 'r')
    superpixel_seg = g_file['node_labeling'][:]
    probas = g_file['edge_feat'][:, 0]  # take initial edge features as weights

    # make sure probas are probas and get a sample prediction
    probas -= probas.min()
    probas /= (probas.max() + 1e-6)
    pred_seg = multicut_from_probas(superpixel_seg, g_file['edges'][:].T, probas)
    pred_seg = torch.from_numpy(pred_seg.astype(np.int64)).to(dev)
    superpixel_seg = torch.from_numpy(superpixel_seg.astype(np.int64)).to(dev)

    # assert the segmentations are consecutive integers
    assert pred_seg.max() == len(torch.unique(pred_seg)) - 1
    assert superpixel_seg.max() == len(torch.unique(superpixel_seg)) - 1

    # add batch dimension
    pred_seg = pred_seg[None]
    superpixel_seg = superpixel_seg[None]

    f = ArtificialCellsReward(sample_shapes)
    rewards = f(pred_seg.long(), superpixel_seg.long())