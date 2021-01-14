from rewards.reward_abc import RewardFunctionAbc
from skimage.measure import approximate_polygon,  find_contours
from skimage.draw import polygon_perimeter
from utils.poly_tangent_space import PolyTangentSpace
import torch

def show_conts(cont, shape, tolerance):
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
        self.gt_turning_functions = []
        for shape_sample in shape_samples:
            shape_sample = shape_sample.cpu().numpy()
            contours = find_contours(shape_sample, level=0)
            for contour in contours:
                poly_approx = torch.from_numpy(approximate_polygon(contour, tolerance=1.2)).to(dev)

                # show_conts(contour, shape_sample.shape, 1.2)

        pass

    def __call__(self, prediction_segmentation, superpixel_segmentation):
        dev = prediction_segmentation.device

        for single_pred, single_sp_seg in zip(prediction_segmentation, superpixel_segmentation):
            # get one-hot representation
            one_hot = torch.zeros((int(single_pred.max()) + 1, ) + single_pred.size(), device=dev)\
                .scatter_(0, single_pred[None], 1)

            # need masses to determine what objects can be considered background
            label_masses = one_hot.flatten(1).sum(-1)
            mask = label_masses >= one_hot.shape[-2] * one_hot.shape[-1] / 4
            bg_ids = torch.nonzero(mask).squeeze(1)  # background label IDs
            object_ids = torch.nonzero(mask == 0).squeeze(1)  # object label IDs

            bg = one_hot[bg_ids]  # get background masks
            objects = one_hot[object_ids]  # get object masks
            bg_sp_ids = torch.unique((single_sp_seg[None] + 1) * bg)[1:] - 1  # mask out the covered superpixels (need to add 1 because the single_sp_seg start from 0)
            object_sp_ids = torch.unique((single_sp_seg[None] + 1) * objects)[1:] - 1

            #TODO get shape descriptors for objects and get a score by comparing to self.descriptors

            #TODO get score for the background

            #TODO project scores from objects to superpixels

            #TODO return scores for each superpixel


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
    rewards = f(pred_seg, superpixel_seg)