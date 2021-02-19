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
from utils.general import calculate_gt_edge_costs
from utils.graphs import run_watershed
import matplotlib.pyplot as plt
from matplotlib import cm
from elf.segmentation.features import compute_rag, compute_affinity_features
from tifffile import imread

tgtdir_train = "/g/kreshuk/hilt/projects/data/toy_lines_ovals/train"
tgtdir_val = "/g/kreshuk/hilt/projects/data/toy_lines_ovals/val"
offs = [[0, -1], [-1, 0], [-1, -1], [-5, 0], [0, -5]]
sep_chnl = 2

def get_heat_map_by_affs(img):
    affs = get_naive_affinities(img[..., np.newaxis], offsets=offs)
    hmap = affs.sum(0) / affs.shape[0]
    return hmap, affs


def preprocess_data():
    for dir in [tgtdir_train, tgtdir_val]:
        pix_dir = os.path.join(dir, 'pix_data')
        graph_dir = os.path.join(dir, 'graph_data')
        fnames_pix = sorted(glob(os.path.join(pix_dir, '*.h5')))
        for i, fname_pix in enumerate(fnames_pix):
            raw = h5py.File(fname_pix, 'r')['raw'][:].squeeze()
            noisy_raw = raw + np.random.normal(0, 0.2, raw.shape)
            noisy_raw = raw
            gt = h5py.File(fname_pix, 'r')['gt'][:]
            head, tail = os.path.split(fname_pix)
            hmap = get_max_hessian_eval(raw, sigma=.5)
            node_labeling = run_watershed(gaussian_filter(hmap, sigma=.1), min_size=10, nhood=4)
            _, affs = get_heat_map_by_affs(gaussian_filter(noisy_raw, sigma=[3.3, 3.3]))
            # node_labeling = run_watershed(gaussian_filter(hmap, sigma=1), min_size=4, nhood=4)
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

            graph_file_name = os.path.join(graph_dir, "graph" + os.path.split(fname_pix)[1][3:])
            graph_file = h5py.File(graph_file_name, 'w')
            # pix_file = h5py.File(os.path.join(pix_dir, "pix_" + str(i) + ".h5"), 'w')

            # pix_file.create_dataset("raw", data=raw, chunks=True)
            # pix_file.create_dataset("gt", data=gt, chunks=True)

            graph_file.create_dataset("edges", data=edges, chunks=True)
            graph_file.create_dataset("offsets", data=np.array(offs), chunks=True)
            graph_file.create_dataset("separating_channel", data=np.array([2]), chunks=True)
            graph_file.create_dataset("edge_feat", data=edge_feat, chunks=True)
            graph_file.create_dataset("diff_to_gt", data=diff_to_gt)
            graph_file.create_dataset("gt_edge_weights", data=gt_edge_weights, chunks=True)
            graph_file.create_dataset("node_labeling", data=node_labeling, chunks=True)
            graph_file.create_dataset("affinities", data=affs, chunks=True)

            graph_file.close()
            # pix_file.close()

    pass


def show_labels(hmap):
    plt.imshow(cm.prism(hmap / hmap.max()));
    plt.show()

if __name__ == "__main__":
    preprocess_data()
    a=1
