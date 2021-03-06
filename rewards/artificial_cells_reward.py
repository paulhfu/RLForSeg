from rewards.reward_abc import RewardFunctionAbc
from skimage.measure import approximate_polygon,  find_contours
from skimage.draw import polygon_perimeter
from utils.polygon_2d import Polygon2d
from cv2 import fitEllipse
import cv2
import torch
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

    def __call__(self, prediction_segmentation, superpixel_segmentation, res, dir_edges, edge_score, *args, **kwargs):
        dev = prediction_segmentation.device
        return_scores = []
        inner_halo_mask = torch.zeros(superpixel_segmentation.shape[1:], device=dev)
        inner_halo_mask[0, :] = 1
        inner_halo_mask[:, 0] = 1
        inner_halo_mask[-1, :] = 1
        inner_halo_mask[:, -1] = 1
        inner_halo_mask = inner_halo_mask.unsqueeze(0)

        for single_pred, single_sp_seg, s_dir_edges in zip(prediction_segmentation, superpixel_segmentation, dir_edges):
            scores = torch.ones(int((single_sp_seg.max()) + 1,), device=dev) * 0.5
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

            if edge_score:
                edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]
                edge_scores = scores[edges].max(dim=0).values
                return_scores.append(edge_scores)
            else:
                return_scores.append(scores)
            #return scores for each superpixel

        return torch.cat(return_scores)


class ArtificialCellsReward2DEllipticFit(RewardFunctionAbc):

    def __init__(self, shape_samples):
        #TODO get the descriptors for the shape samples
        dev = shape_samples.device
        self.gt_descriptors = []
        widths, heights = [], []
        for mask in shape_samples:
            cnt = find_contours(mask.cpu().numpy(), level=0)[0]

            # img = torch.zeros_like(wtsd[:, slc, :]).cpu()
            # img[cnt[:, 0], cnt[:, 1]] = 1
            # plt.imshow(img);plt.show()

            ellipseT = fitEllipse(cnt.astype(np.int))
            widths.append(ellipseT[1][1])
            heights.append(ellipseT[1][0])
        self.expected_width = np.array(widths).mean()
        self.expected_height = np.array(heights).mean()
        self.expected_ratio = self.expected_width / self.expected_height


    def __call__(self, prediction_segmentation, superpixel_segmentation, res, dir_edges, edge_score, *args, **kwargs):
        dev = prediction_segmentation.device
        return_scores = []

        for single_pred, single_sp_seg, s_dir_edges in zip(prediction_segmentation, superpixel_segmentation, dir_edges):
            scores = torch.ones(int((single_sp_seg.max()) + 1,), device=dev) * 0.5
            if single_pred.max() == 0:  # image is empty
                return_scores.append(scores - 0.5)
                continue
            # get one-hot representation. One mask for each object in the predictions
            one_hot = torch.zeros((int(single_pred.max()) + 1, ) + single_pred.size(), device=dev, dtype=torch.long) \
                .scatter_(0, single_pred[None], 1)

            # need masses to determine what objects can be considered background
            label_masses = one_hot.flatten(1).sum(-1)
            bg_mask = torch.zeros_like(label_masses).bool()  # prepare a label for the background
            bg_id = label_masses.argmax()  # determine largest object (is suspected to be background)
            bg_mask[bg_id] = True  # set the prospective bg labels to true
            false_obj_mask = (label_masses < 15) | (label_masses > 50**2)  # if masses are too large or too small we can be sure these are false objects
            false_obj_mask[bg_id] = False
            # everything else are potential objects
            potenial_obj_mask = (false_obj_mask == False)
            potenial_obj_mask[bg_id] = False
            potential_object_ids = torch.nonzero(potenial_obj_mask).squeeze(1)  # object label IDs

            bg = one_hot[bg_id]  # get background masks
            objects = one_hot[potential_object_ids]  # get object masks
            # get the label IDs for the object classes
            false_obj_sp_ids = torch.unique((single_sp_seg[None] + 1) * one_hot[false_obj_mask])[1:] - 1
            object_sp_ids = [torch.unique((single_sp_seg[None] + 1) * obj)[1:] - 1 for obj in objects]

            # iterate over all eventual objects
            for object, obj_id, sp_ids in zip(objects, potential_object_ids, object_sp_ids):
                try:
                    contour = find_contours(object.cpu().numpy(), level=0)
                    if len(contour) > 1:
                        scores[sp_ids] -= 0.1
                        continue
                    contour = contour[0]
                except Exception as e:
                    print(e)
                    scores[sp_ids] -= 0.1
                    continue
                poly_chain = torch.from_numpy(approximate_polygon(contour, tolerance=0.0)).to(dev)  # get the polygonal chain from the contour (contour might be very long,polygonal chain removes unnecessary points)
                if poly_chain.shape[0] <= 5:  # check if this might be a valid object
                    print("WARNING polychain too small")
                    scores[sp_ids] -= 0.1
                    continue
                score = 0

                try:
                    ellipseT = fitEllipse(contour.astype(np.int))
                    ratio = ellipseT[1][1] / (ellipseT[1][0] + 1e-10)  # calculate the reatio of the ellipsis two main axes
                    ratio_diff = abs(self.expected_ratio - ratio)  # see how much this differs from our expected ratio
                    len_diff = abs(self.expected_width - ellipseT[1][1]) + abs(ellipseT[1][0] - self.expected_height)  # check each axis for themselves

                    if len_diff < (self.expected_width + self.expected_height):  # according to the scale of the diff we penalize differently
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
                # plt.imshow(object.cpu()[..., None] * 200 + cv2.ellipse(np.zeros(shape=[256, 256, 3], dtype=np.uint8), ellipse, (23, 184, 80), 3))
                # plt.show()

                scores[sp_ids] += score  # assign our calculated scores to each superpixel label id that is in the object
            if len(false_obj_sp_ids) > 0:  # penalize all objects which have the wrong mass
                scores[false_obj_sp_ids] -= 0.5
            if edge_score:  # if we want a score per edge we do the tf from sp to edges
                edges = s_dir_edges[:, :int(s_dir_edges.shape[1] / 2)]  # for each undirected edge get the scores of the two incidental superpixels
                edge_scores = scores[edges].max(dim=0).values  # from this two scores select the max
                return_scores.append(edge_scores)
            else:
                return_scores.append(scores)
        return torch.cat(return_scores)  # collect all scores

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