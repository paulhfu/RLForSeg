import os

from glob import glob
import h5py
import numpy as np
import torch
from skimage import draw
from scipy.ndimage import gaussian_filter
import elf
import nifty

from affogato.segmentation import compute_mws_segmentation
from utils.affinities import get_naive_affinities, get_edge_features_1d, get_max_hessian_eval, get_hessian_det
from utils.general import calculate_gt_edge_costs, random_label_cmap
from utils.graphs import run_watershed
import matplotlib.pyplot as plt
from matplotlib import cm
from elf.segmentation.features import compute_rag, compute_affinity_features
from tifffile import imread

tgtdir_train = "/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/train"
tgtdir_val = "/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/val"
offs = [[1, 0], [0, 1], [2, 0], [0, 2], [3, 0], [0, 3], [4, 0], [0, 4], [8, 0], [0, 8], [16, 0], [0, 16]]
sep_chnl = 2

def get_data(img, gt, affs, sigma, strides=[4, 4], overseg_factor=1.2, random_strides=False, fname='ex1.h5'):
    affinities = affs.copy()
    affinities[:sep_chnl] /= overseg_factor
    # affinities[sep_chnl:] *= overseg_factor
    # affinities = np.clip(affinities, 0, 1)

    # scale affinities in order to get an oversegmentation
    affinities[:sep_chnl] /= overseg_factor
    node_labeling = compute_mws_segmentation(affinities, offs, sep_chnl, strides=strides, randomize_strides=random_strides)
    node_labeling = node_labeling - 1
    nodes = np.unique(node_labeling)
    save_file = h5py.File('/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/train/raw_wtsd_cpy/exs/' + fname, 'w')
    save_file.create_dataset(name='data', data=node_labeling)
    save_file.close()
    plt.imshow(cm.prism(node_labeling / node_labeling.max()));
    plt.show()
    # try:
    #     assert all(nodes == np.array(range(len(nodes)), dtype=np.float))
    # except:
    #     Warning("node ids are off")
    #
    # # get edges from node labeling and edge features from affinity stats
    # edge_feat, neighbors = get_edge_features_1d(node_labeling, offs, affinities)
    # # get gt edge weights based on edges and gt image
    # gt_edge_weights = calculate_gt_edge_costs(neighbors, node_labeling.squeeze(), gt.squeeze())
    # edges = neighbors.astype(np.long)
    #
    # # calc multicut from gt
    # gt_seg = multicut_from_probas(node_labeling, edges, gt_edge_weights)
    #
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    # ax1.imshow(cm.prism(gt/gt.max()));ax1.set_title('gt')
    # ax2.imshow(cm.prism(node_labeling / node_labeling.max()));ax2.set_title('sp')
    # ax3.imshow(cm.prism(gt_seg / gt_seg.max()));ax3.set_title('mc')
    # ax4.imshow(img);ax4.set_title('raw')
    # plt.show()
    #
    # affinities = affinities.astype(np.float32)
    # edge_feat = edge_feat.astype(np.float32)
    # nodes = nodes.astype(np.float32)
    # node_labeling = node_labeling.astype(np.float32)
    # gt_edge_weights = gt_edge_weights.astype(np.float32)
    # diff_to_gt = np.abs((edge_feat[:, 0] - gt_edge_weights)).sum()
    #
    # edges = np.sort(edges, axis=-1)
    # edges = edges.T
    #
    # return img, gt, edges, edge_feat, diff_to_gt, gt_edge_weights, node_labeling, nodes, affinities

def preprocess_data_1():
    for dir in [tgtdir_train, tgtdir_val]:
        fnames = sorted(glob(os.path.join(dir, 'raw_wtsd/*.h5')))
        pix_dir = os.path.join(dir, 'pix_data')
        graph_dir = os.path.join(dir, 'graph_data')
        for i, fname in enumerate(fnames):
            raw = h5py.File(fname, 'r')['raw'][:]
            gt = h5py.File(fname, 'r')['wtsd'][:]
            head, tail = os.path.split(fname)
            hmap = torch.from_numpy(h5py.File(os.path.join(dir, 'edt', tail[:-3] + '_predictions' + '.h5'), 'r')['predictions'][:]).squeeze()
            hmap = torch.sigmoid(hmap).numpy()
            # sep = affs.shape[0] // 2
            # affs = torch.sigmoid(affs)

            node_labeling = run_watershed(gaussian_filter(hmap, sigma=.2), min_size=4)
            edge_feat, edges = get_edge_features_1d(node_labeling, offs, affs)
            gt_edge_weights = calculate_gt_edge_costs(edges, node_labeling.squeeze(), gt.squeeze())

            edges = edges.astype(np.long)

            affs = affs.astype(np.float32)
            edge_feat = edge_feat.astype(np.float32)
            node_labeling = node_labeling.astype(np.float32)
            gt_edge_weights = gt_edge_weights.astype(np.float32)
            diff_to_gt = np.abs((edge_feat[:, 0] - gt_edge_weights)).sum()
            edges = np.sort(edges, axis=-1)
            edges = edges.T

            graph_file = h5py.File(os.path.join(graph_dir, "graph_" + str(i) + ".h5"), 'w')
            pix_file = h5py.File(os.path.join(pix_dir, "pix_" + str(i) + ".h5"), 'w')

            pix_file.create_dataset("raw", data=raw, chunks=True)
            pix_file.create_dataset("gt", data=gt, chunks=True)

            graph_file.create_dataset("edges", data=edges, chunks=True)
            graph_file.create_dataset("edge_feat", data=edge_feat, chunks=True)
            graph_file.create_dataset("diff_to_gt", data=diff_to_gt)
            graph_file.create_dataset("gt_edge_weights", data=gt_edge_weights, chunks=True)
            graph_file.create_dataset("node_labeling", data=node_labeling, chunks=True)
            graph_file.create_dataset("affinities", data=affs, chunks=True)

            graph_file.close()
            pix_file.close()

    pass

def preprocess_data():
    for dir in [tgtdir_train, tgtdir_val]:
        fnames = sorted(glob(os.path.join(dir, 'raw_wtsd/*.h5')))
        pix_dir = os.path.join(dir, 'pix_data')
        graph_dir = os.path.join(dir, 'graph_data')
        for i, fname in enumerate(fnames):
            raw = h5py.File(fname, 'r')['raw'][:]
            gt = h5py.File(fname, 'r')['wtsd'][:]
            head, tail = os.path.split(fname)
            affs = torch.from_numpy(h5py.File(os.path.join(dir, 'affinities_1', tail[:-3] + '_predictions' + '.h5'), 'r')['predictions'][:]).squeeze(1)
            affs = torch.sigmoid(affs).numpy()
            # sep = affs.shape[0] // 2
            # affs = torch.sigmoid(affs)

            node_labeling = run_watershed(gaussian_filter(affs[0] + affs[1] + affs[2] + affs[3], sigma=.2), min_size=4)

            # relabel to consecutive ints starting at 0
            node_labeling = torch.from_numpy(node_labeling.astype(np.long))
            gt = torch.from_numpy(gt.astype(np.long))
            mask = node_labeling[None] == torch.unique(node_labeling)[:, None, None]
            node_labeling = (mask * (torch.arange(len(torch.unique(node_labeling)), device=node_labeling.device)[:, None, None] + 1)).sum(
                0) - 1

            mask = gt[None] == torch.unique(gt)[:, None, None]
            gt = (mask * (torch.arange(len(torch.unique(gt)), device=gt.device)[:, None, None] + 1)).sum(0) - 1

            edge_feat, edges = get_edge_features_1d(node_labeling.numpy(), offs, affs)
            gt_edge_weights = calculate_gt_edge_costs(torch.from_numpy(edges.astype(np.long)), node_labeling.squeeze(), gt.squeeze(), 0.5)

            gt_edge_weights = gt_edge_weights.numpy()
            gt = gt.numpy()
            node_labeling = node_labeling.numpy()
            edges = edges.astype(np.long)

            affs = affs.astype(np.float32)
            edge_feat = edge_feat.astype(np.float32)
            node_labeling = node_labeling.astype(np.float32)
            gt_edge_weights = gt_edge_weights.astype(np.float32)
            diff_to_gt = np.abs((edge_feat[:, 0] - gt_edge_weights)).sum()
            edges = np.sort(edges, axis=-1)
            edges = edges.T

            graph_file = h5py.File(os.path.join(graph_dir, "graph_" + str(i) + ".h5"), 'w')
            pix_file = h5py.File(os.path.join(pix_dir, "pix_" + str(i) + ".h5"), 'w')

            pix_file.create_dataset("raw", data=raw, chunks=True)
            pix_file.create_dataset("gt", data=gt, chunks=True)

            graph_file.create_dataset("edges", data=edges, chunks=True)
            graph_file.create_dataset("edge_feat", data=edge_feat, chunks=True)
            graph_file.create_dataset("diff_to_gt", data=diff_to_gt)
            graph_file.create_dataset("gt_edge_weights", data=gt_edge_weights, chunks=True)
            graph_file.create_dataset("node_labeling", data=node_labeling, chunks=True)
            graph_file.create_dataset("affinities", data=affs, chunks=True)

            graph_file.close()
            pix_file.close()

    pass

def transfer_to_slices_in_files():
    # source_file_wtsd = "/g/kreshuk/data/leptin/sourabh_data_v1/Segmentation_results_fused_tp_1_ch_0_Masked_WatershedBoundariesMergeTreeFilter_Out1.tif"
    # source_file_raw = "/g/kreshuk/data/leptin/sourabh_data_v1/Original_fused_tp_1_ch_0.tif"
    # raw = np.array(imread(source_file_raw))
    # wtsd = np.array(imread(source_file_wtsd))
    raw = h5py.File('/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/raw.h5', "r")["data"][:]

    # wtsd = h5py.File('/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/Masked_WatershedBoundariesMergeTreeFilter_Out1.h5', "r")["data"][:]
    # hmap = h5py.File('/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/hmap.h5', "r")["data"][:]

    hmap = get_max_hessian_eval(gaussian_filter(raw, sigma=[2.3, 2.3, 2.3]), sigma=[1, 1, 1])
    file = h5py.File('/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/hmap_smoothed_max_hess_eval_1.h5', "w")
    file.create_dataset(name='data', data=hmap)
    file.close()

    hmap = get_max_hessian_eval(gaussian_filter(raw, sigma=[3.3, 3.3, 3.3]), sigma=[1, 1, 1])
    file = h5py.File('/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/hmap_smoothed_max_hess_eval_2.h5', "w")
    file.create_dataset(name='data', data=hmap)
    file.close()

    # dir = os.path.join(tgtdir_train, "raw_wtsd")
    #
    # for i in range(raw.shape[1]):
    #     raw_file = h5py.File(os.path.join(dir, "slice_" + str(i) + ".h5"), 'w')
    #
    #     raw_file.create_dataset("raw", data=raw[:, i, :], chunks=True)
    #     raw_file.create_dataset("wtsd", data=wtsd[:, i, :], chunks=True)
    #
    #     raw_file.close()

def show_labels(hmap):
    plt.imshow(cm.prism(hmap / hmap.max()));
    plt.show()

if __name__ == "__main__":

    # transfer_to_slices_in_files()
    preprocess_data()
    a=1
