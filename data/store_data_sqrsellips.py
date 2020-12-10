import os

import h5py
import numpy as np
from skimage import draw
from skimage.filters import gaussian
import elf
import nifty

from affogato.segmentation import compute_mws_segmentation
from utils.affinities import get_naive_affinities, get_edge_features_1d
from utils.general import calculate_gt_edge_costs, set_seed_everywhere
from data.spg_dset import SpgDset
import matplotlib.pyplot as plt
from matplotlib import cm

# set_seed_everywhere(19)


def get_pix_data(shape=(256, 256)):
    """ This generates raw-gt-superpixels and correspondinng rags of rectangles and circles"""

    edge_offsets = [[0, -1], [-1, 0], [-3, 0], [0, -3], [-6, 0], [0, -6]]  # offsets defining the edges for pixel affinities
    overseg_factor = 1.8
    sep_chnl = 2  # channel separating attractive from repulsive edges
    n_circles = 5  # number of ellipses in image
    n_polys = 10  # number of rand polys in image
    n_rect = 5  # number rectangles in image
    circle_color = np.array([1, 0, 0], dtype=np.float)
    rect_color = np.array([0, 0, 1], dtype=np.float)
    col_diff = 0.4  # by this margin object color can vary ranomly
    min_r, max_r = 10, 20  # min and max radii of ellipses/circles
    min_dist = max_r

    img = np.random.randn(*(shape + (3,))) / 5  # init image with some noise
    gt = np.zeros(shape)

    #  get some random frequencies
    ri1, ri2, ri3, ri4, ri5, ri6 = np.sign(np.random.randint(-100, 100)) * ((np.random.rand() * 2) + .5), np.sign(
        np.random.randint(-100, 100)) * ((np.random.rand() * 2) + .5), (np.random.rand() * 4) + 3, (
                                           np.random.rand() * 4) + 3, np.sign(np.random.randint(-100, 100)) * (
                                           (np.random.rand() * 2) + .5), np.sign(np.random.randint(-100, 100)) * (
                                               (np.random.rand() * 2) + .5)
    x = np.zeros(shape)
    x[:, :] = np.arange(img.shape[0])[np.newaxis, :]
    y = x.transpose()
    # add background frequency interferences
    img += (np.sin(np.sqrt((x * ri1) ** 2 + ((shape[1] - y) * ri2) ** 2) * ri3 * np.pi / shape[0]))[
        ..., np.newaxis]
    img += (np.sin(np.sqrt((x * ri5) ** 2 + ((shape[1] - y) * ri6) ** 2) * ri4 * np.pi / shape[1]))[
        ..., np.newaxis]
    # smooth a bit
    img = gaussian(np.clip(img, 0.1, 1), sigma=.8)
    # add some circles
    circles = []
    cmps = []
    while len(circles) < n_circles:
        mp = np.random.randint(min_r, shape[0] - min_r, 2)
        too_close = False
        for cmp in cmps:
            if np.linalg.norm(cmp - mp) < min_dist:
                too_close = True
        if too_close:
            continue
        r = np.random.randint(min_r, max_r, 2)
        circles.append(draw.circle(mp[0], mp[1], r[0], shape=shape))
        cmps.append(mp)

    # add some random polygons
    polys = []
    while len(polys) < n_polys:
        mp = np.random.randint(min_r, shape[0] - min_r, 2)
        too_close = False
        for cmp in cmps:
            if np.linalg.norm(cmp - mp) < min_dist // 2:
                too_close = True
        if too_close:
            continue
        circle = draw.circle_perimeter(mp[0], mp[1], max_r)
        poly_vert = np.random.choice(len(circle[0]), np.random.randint(3, 6), replace=False)
        polys.append(draw.polygon(circle[0][poly_vert], circle[1][poly_vert], shape=shape))
        cmps.append(mp)

    # add some random rectangles
    rects = []
    while len(rects) < n_rect:
        mp = np.random.randint(min_r, shape[0] - min_r, 2)
        _len = np.random.randint(min_r // 2, max_r, (2,))
        too_close = False
        for cmp in cmps:
            if np.linalg.norm(cmp - mp) < min_dist:
                too_close = True
        if too_close:
            continue
        start = (mp[0] - _len[0], mp[1] - _len[1])
        rects.append(draw.rectangle(start, extent=(_len[0] * 2, _len[1] * 2), shape=shape))
        cmps.append(mp)


    # draw polys and give them some noise
    for poly in polys:
        color = np.random.rand(3)
        while np.linalg.norm(color - circle_color) < col_diff or np.linalg.norm(
                color - rect_color) < col_diff:
            color = np.random.rand(3)
        img[poly[0], poly[1], :] = color
        img[poly[0], poly[1], :] += np.random.randn(len(poly[1]), 3) / 5  # add noise to the polygons

    # draw circles with some frequency
    cols = np.random.choice(np.arange(4, 11, 1).astype(np.float) / 10, n_circles, replace=False)
    for i, circle in enumerate(circles):
        gt[circle[0], circle[1]] = 1 + (i/10)
        ri1, ri2, ri3, ri4, ri5, ri6 = np.sign(np.random.randint(-100, 100)) * ((np.random.rand() * 4) + 7), np.sign(
            np.random.randint(-100, 100)) * ((np.random.rand() * 4) + 7), (np.random.rand() + 1) * 3, (
                                               np.random.rand() + 1) * 3, np.sign(np.random.randint(-100, 100)) * (
                                                   (np.random.rand() * 4) + 7), np.sign(
            np.random.randint(-100, 100)) * ((np.random.rand() * 4) + 7)
        img[circle[0], circle[1], :] = np.array([cols[i], 0.0, 0.0])
        img[circle[0], circle[1], :] += np.array([1.0, 1.0, 0.0]) * ((np.sin(np.sqrt(
            (x[circle[0], circle[1]] * ri5) ** 2 + (
                        (shape[1] - y[circle[0], circle[1]]) * ri2) ** 2) * ri3 * np.pi / shape[0]))[
                                                                           ..., np.newaxis] * 0.15) + 0.2
        img[circle[0], circle[1], :] += np.array([1.0, 1.0, 0.0]) * ((np.sin(np.sqrt(
            (x[circle[0], circle[1]] * ri6) ** 2 + (
                        (shape[1] - y[circle[0], circle[1]]) * ri1) ** 2) * ri4 * np.pi / shape[1]))[
                                                                           ..., np.newaxis] * 0.15) + 0.2

    # draw rectangles with some frequency
    cols = np.random.choice(np.arange(4, 11, 1).astype(np.float) / 10, n_rect, replace=False)
    for i, rect in enumerate(rects):
        gt[rect[0], rect[1]] = 2+(i/10)
        ri1, ri2, ri3, ri4, ri5, ri6 = np.sign(np.random.randint(-100, 100)) * ((np.random.rand() * 4) + 7), np.sign(
            np.random.randint(-100, 100)) * ((np.random.rand() * 4) + 7), (np.random.rand() + 1) * 3, (
                                               np.random.rand() + 1) * 3, np.sign(np.random.randint(-100, 100)) * (
                                                   (np.random.rand() * 4) + 7), np.sign(
            np.random.randint(-100, 100)) * ((np.random.rand() * 4) + 7)
        img[rect[0], rect[1], :] = np.array([0.0, 0.0, cols[i]])
        img[rect[0], rect[1], :] += np.array([1.0, 1.0, 0.0]) * ((np.sin(
            np.sqrt((x[rect[0], rect[1]] * ri5) ** 2 + ((shape[1] - y[rect[0], rect[1]]) * ri2) ** 2) * ri3 * np.pi /
            shape[0]))[..., np.newaxis] * 0.15) + 0.2
        img[rect[0], rect[1], :] += np.array([1.0, 1.0, 0.0]) * ((np.sin(
            np.sqrt((x[rect[0], rect[1]] * ri1) ** 2 + ((shape[1] - y[rect[0], rect[1]]) * ri6) ** 2) * ri4 * np.pi /
            shape[1]))[..., np.newaxis] * 0.15) + 0.2

    img = np.clip(img, 0, 1)  # clip to valid range
    # get affinities and calc superpixels with mutex watershed
    affinities = get_naive_affinities(gaussian(np.clip(img, 0, 1), sigma=.2), edge_offsets)
    affinities[:sep_chnl] *= -1
    affinities[:sep_chnl] += +1
    # scale affinities in order to get an oversegmentation
    affinities[:sep_chnl] /= overseg_factor
    affinities[sep_chnl:] *= overseg_factor
    affinities = np.clip(affinities, 0, 1)
    node_labeling = compute_mws_segmentation(affinities, edge_offsets, sep_chnl)
    node_labeling = node_labeling - 1
    nodes = np.unique(node_labeling)
    try:
        assert all(nodes == np.array(range(len(nodes)), dtype=np.float))
    except:
        Warning("node ids are off")

    # get edges from node labeling and edge features from affinity stats
    edge_feat, neighbors = get_edge_features_1d(node_labeling, edge_offsets, affinities)
    # get gt edge weights based on edges and gt image
    gt_edge_weights = calculate_gt_edge_costs(neighbors, node_labeling.squeeze(), gt.squeeze())
    edges = neighbors.astype(np.long)

    # calc multicut from gt
    gt_seg = get_current_soln(gt_edge_weights, node_labeling, edges)
    # show result (uncomment for testing)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.imshow(cm.prism(gt/gt.max()));ax1.set_title('gt')
    ax2.imshow(cm.prism(node_labeling / node_labeling.max()));ax2.set_title('sp')
    ax3.imshow(cm.prism(gt_seg / gt_seg.max()));ax3.set_title('mc')
    ax4.imshow(img);ax4.set_title('raw')
    plt.show()

    affinities = affinities.astype(np.float32)
    edge_feat = edge_feat.astype(np.float32)
    nodes = nodes.astype(np.float32)
    node_labeling = node_labeling.astype(np.float32)
    gt_edge_weights = gt_edge_weights.astype(np.float32)
    diff_to_gt = np.abs((edge_feat[:, 0] - gt_edge_weights)).sum()

    edges = np.sort(edges, axis=-1)
    edges = edges.T

    return img, gt, edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, nodes, affinities


def get_current_soln(edge_weights, sp_seg, edge_ids):
    p_min = 0.001
    p_max = 1.
    probs = edge_weights
    # probs = self.b_gt_edge_weights[self.e_offs[i-1]:self.e_offs[i]]
    edges = edge_ids
    costs = (p_max - p_min) * probs + p_min
    # probabilities to costs
    costs = np.log((1. - costs) / costs)
    graph = nifty.graph.undirectedGraph(len(np.unique(sp_seg)))
    graph.insertEdges(edges)

    node_labels = elf.segmentation.multicut.multicut_kernighan_lin(graph, costs)

    mc_seg = np.zeros_like(sp_seg)
    for j, lbl in enumerate(node_labels):
        mc_seg += (sp_seg == j).astype(np.uint) * lbl

    return mc_seg


def store_all(base_dir):
    pix_dir = os.path.join(base_dir, 'pix_data')
    graph_dir = os.path.join(base_dir, 'graph_data')


    for i in range(1):
        raw, gt, edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, nodes, affinities = get_pix_data()

        graph_file = h5py.File(os.path.join(graph_dir, "graph_" + str(i) + ".h5"), 'w')
        pix_file = h5py.File(os.path.join(pix_dir, "pix_" + str(i) + ".h5"), 'w')

        pix_file.create_dataset("raw", data=raw, chunks=True)
        pix_file.create_dataset("gt", data=gt, chunks=True)

        graph_file.create_dataset("edges", data=edges, chunks=True)
        graph_file.create_dataset("edge_feat", data=edge_feat, chunks=True)
        graph_file.create_dataset("diff_to_gt", data=diff_to_gt)
        graph_file.create_dataset("gt_edge_weights", data=gt_edge_weights, chunks=True)
        graph_file.create_dataset("node_labeling", data=node_labeling, chunks=True)
        graph_file.create_dataset("affinities", data=affinities, chunks=True)

        graph_file.close()
        pix_file.close()


if __name__ == "__main__":
    dir = "/g/kreshuk/hilt/projects/fewShotLearning/mutexWtsd/data/storage/sqrs_crclspn/pix_and_graphs"
    get_pix_data()

    dset = SpgDset(dir)
