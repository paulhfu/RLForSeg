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
from utils.general import calculate_gt_edge_costs, random_label_cmap, multicut_from_probas
from utils.graphs import run_watershed
from models.feature_extractor import FeExtractor
import matplotlib.pyplot as plt
from matplotlib import cm
from elf.segmentation.features import compute_rag, compute_affinity_features
from tifffile import imread
from utils.distances import CosineDistance
from utils.pt_gaussfilter import GaussianSmoothing
from utils.affinities import get_affinities_from_embeddings_2d
import threading
import torch.nn.functional as F
tgtdir_train = "/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/train"
tgtdir_val = "/g/kreshuk/hilt/projects/data/leptin_fused"
offs = [[1, 0], [0, 1], [2, 0], [0, 2]]
# offs = [[1, 0], [0, 1], [2, 0], [0, 2], [3, 0], [0, 3], [4, 0], [0, 4], [8, 0], [0, 8], [16, 0], [0, 16]]
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

tgtdir_train = "/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/train"
tgtdir_val = "/g/kreshuk/hilt/projects/data/leptin_fused_tp1_ch_0/true_val"

backbone = {
    "name": "UNet2D",
    "in_channels": 2,
    "out_channels": 16,
    "layer_order": "bcr",
    "num_groups": 8,
    "f_maps": [32, 64, 128, 256],
    "final_sigmoid": False,
    "is_segmentation": False
}

def preprocess_data():
    gauss_kernel = GaussianSmoothing(1, 5, 3, device="cpu")
    distance = CosineDistance()
    device = "cuda:0"
    fe_model_name = "/g/kreshuk/hilt/storage/leptin_data_nets/best_val_model.pth"

    model = FeExtractor(backbone, distance, device)
    model.embed_model.load_state_dict(torch.load(fe_model_name))
    model.cuda("cuda:0")

    offs = [[1, 0], [0, 1], [2, 0], [0, 2], [4, 0], [0, 4], [8, 0], [0, 8], [16, 0], [0, 16]]
    for j, dir in enumerate([tgtdir_val]):
        pix_dir = os.path.join(dir, 'pix_data')
        graph_dir = os.path.join(dir, 'graph_data')
        new_pix_dir = os.path.join(dir, "bg_masked_data", 'pix_data')
        new_graph_dir = os.path.join(dir, "bg_masked_data", 'graph_data')
        fnames = sorted(glob(os.path.join(pix_dir, '*.h5')))
        def process_file(i):
            fname = fnames[i]
            head, tail = os.path.split(fname)
            num = tail[4:-3]
            # os.rename(os.path.join(graph_dir, "graph_" + str(i) + ".h5"), os.path.join(graph_dir, "graph_" + num + ".h5"))
            # os.rename(os.path.join(pix_dir, "pix_" + str(i) + ".h5"), os.path.join(pix_dir, "pix_" + num + ".h5"))

            # raw = torch.from_numpy(h5py.File(fname, 'r')['raw'][:].astype(np.float))
            # gt = h5py.File(fname, 'r')['label'][:].astype(np.long)
            # affs = torch.from_numpy(h5py.File(os.path.join(dir, 'affinities', tail[:-3] + '_predictions' + '.h5'), 'r')['predictions'][:]).squeeze(1)


            # raw -= raw.min()
            # raw /= raw.max()
            #
            # node_labeling = run_watershed(gaussian_filter(affs[0] + affs[1] + affs[2] + affs[3], sigma=.2), min_size=4)
            #
            # # relabel to consecutive ints starting at 0
            # node_labeling = torch.from_numpy(node_labeling.astype(np.long))
            #
            #
            # gt = torch.from_numpy(gt.astype(np.long))
            # mask = node_labeling[None] == torch.unique(node_labeling)[:, None, None]
            # node_labeling = (mask * (torch.arange(len(torch.unique(node_labeling)), device=node_labeling.device)[:, None, None] + 1)).sum(
            #     0) - 1
            #
            # node_labeling += 2
            # # bgm
            # node_labeling[gt == 0] = 0
            # node_labeling[gt == 1] = 1
            # plt.imshow(node_labeling);plt.show()
            #
            # mask = gt[None] == torch.unique(gt)[:, None, None]
            # gt = (mask * (torch.arange(len(torch.unique(gt)), device=gt.device)[:, None, None] + 1)).sum(0) - 1
            #
            edge_img = get_contour_from_2d_binary(node_labeling[None, None].float())
            # edge_img = gauss_kernel(edge_img.float())
            # raw = torch.cat([raw[None, None], edge_img], dim=1).squeeze(0).numpy()
            # affs = torch.sigmoid(affs).numpy()
            #
            # edge_feat, edges = get_edge_features_1d(node_labeling.numpy(), offs, affs[:4])
            #
            # gt_edge_weights = gt_edge_weights.numpy()
            # gt = gt.numpy()
            # node_labeling = node_labeling.numpy()
            # edges = edges.astype(np.long)
            #
            # affs = affs.astype(np.float32)
            # edge_feat = edge_feat.astype(np.float32)
            # node_labeling = node_labeling.astype(np.float32)
            # gt_edge_weights = gt_edge_weights.astype(np.float32)
            # diff_to_gt = np.abs((edge_feat[:, 0] - gt_edge_weights)).sum()
            # edges = np.sort(edges, axis=-1)
            # edges = edges.T
            # #
            # #
            graph_file = h5py.File(os.path.join(graph_dir, "graph_" + num + ".h5"), 'r')
            pix_file = h5py.File(os.path.join(pix_dir, "pix_" + num + ".h5"), 'r')

            raw = pix_file["raw_2chnl"][:]
            gt = pix_file["gt"][:]
            sp_seg = graph_file["node_labeling"][:]
            edges = graph_file["edges"][:]
            affs = graph_file["affinities"][:]

            graph_file.close()
            pix_file.close()
            # plt.imshow(multicut_from_probas(sp_seg, edges.T, calculate_gt_edge_costs(torch.from_numpy(edges.T.astype(np.long)).to(dev), torch.from_numpy(sp_seg).to(dev), torch.from_numpy(gt.squeeze()).to(dev), 1.5).cpu()), cmap=random_label_cmap(), interpolation="none");plt.show()
            with torch.set_grad_enabled(False):
                embeddings = model(torch.from_numpy(raw).to(device).float()[None])
            emb_affs = get_affinities_from_embeddings_2d(embeddings, offs, 0.4, distance)[0].cpu().numpy()
            ew_embedaffs = 1 - get_edge_features_1d(sp_seg, offs, emb_affs)[0][:, 0]
            mc_soln = torch.from_numpy(multicut_from_probas(sp_seg, edges.T, ew_embedaffs).astype(np.long)).to(device)

            mask = mc_soln[None] == torch.unique(mc_soln)[:, None, None]
            mc_soln = (mask * (torch.arange(len(torch.unique(mc_soln)), device=device)[:, None, None] + 1)).sum(0) - 1

            masses = (mc_soln[None] == torch.unique(mc_soln)[:, None, None]).sum(-1).sum(-1)
            bg1id = masses.argmax()
            masses[bg1id] = 0
            bg1_mask = mc_soln == bg1id
            bg2_mask = mc_soln == masses.argmax()

            sp_seg = torch.from_numpy(sp_seg.astype(np.long)).to(device)

            mask = sp_seg[None] == torch.unique(sp_seg)[:, None, None]
            sp_seg = (mask * (torch.arange(len(torch.unique(sp_seg)), device=sp_seg.device)[:, None, None] + 1)).sum(0) - 1

            sp_seg += 2
            sp_seg *= (bg1_mask == 0)
            sp_seg *= (bg2_mask == 0)
            sp_seg += bg2_mask

            mask = sp_seg[None] == torch.unique(sp_seg)[:, None, None]
            sp_seg = (mask * (torch.arange(len(torch.unique(sp_seg)), device=sp_seg.device)[:, None, None] + 1)).sum(0) - 1
            sp_seg = sp_seg.cpu()

            raw -= raw.min()
            raw /= raw.max()
            edge_feat, edges = get_edge_features_1d(sp_seg.numpy(), offs[:4], affs[:4])
            edges = edges.astype(np.long)

            gt_edge_weights = calculate_gt_edge_costs(torch.from_numpy(edges).to(device), sp_seg.to(device), torch.from_numpy(gt).to(device), 0.4)
            node_labeling = sp_seg.numpy()

            affs = affs.astype(np.float32)
            edge_feat = edge_feat.astype(np.float32)
            node_labeling = node_labeling.astype(np.float32)
            gt_edge_weights = gt_edge_weights.cpu().numpy().astype(np.float32)
            # edges = np.sort(edges, axis=-1)
            edges = edges.T
            # plt.imshow(sp_seg.cpu(), cmap=random_label_cmap(), interpolation="none");plt.show()
            # plt.imshow(bg1_mask.cpu());plt.show()
            # plt.imshow(bg2_mask.cpu());plt.show()
            new_pix_file = h5py.File(os.path.join(new_pix_dir, "pix_" + num + ".h5"), 'w')
            new_graph_file = h5py.File(os.path.join(new_graph_dir, "graph_" + num + ".h5"), 'w')

            # plt.imshow(gt_sp_projection, cmap=random_label_cmap(), interpolation="none");plt.show()

            new_pix_file.create_dataset("raw_2chnl", data=raw, chunks=True)
            new_pix_file.create_dataset("gt", data=gt, chunks=True)
            #
            new_graph_file.create_dataset("edges", data=edges, chunks=True)
            new_graph_file.create_dataset("edge_feat", data=edge_feat, chunks=True)
            new_graph_file.create_dataset("gt_edge_weights", data=gt_edge_weights, chunks=True)
            new_graph_file.create_dataset("node_labeling", data=node_labeling, chunks=True)
            new_graph_file.create_dataset("affinities", data=affs, chunks=True)
            new_graph_file.create_dataset("offsets", data=np.array([[1, 0], [0, 1], [2, 0], [0, 2], [4, 0], [0, 4], [8, 0], [0, 8], [16, 0], [0, 16]]), chunks=True)
            #
            new_graph_file.close()
            new_pix_file.close()

        workers = []
        for i in range(len(fnames)):
            worker = threading.Thread(target=process_file, args=(i,))
            worker.start()
            workers.append(worker)

        for worker in workers:
            worker.join()


    pass

def check_sp():

    dir = "/g/kreshuk/kaziakhm/leptin_data/train/confocal_2D_unet_bce_dice_ds2x"
    fnames = sorted(glob(os.path.join(dir, '*.h5')))
    pix_dir = os.path.join(dir, "bg_masked_data", 'pix_data')
    graph_dir = os.path.join(dir, "bg_masked_data", 'graph_data')
    for fname in fnames:
        edge_pred = h5py.File(fname, 'r')["predictions"][:].squeeze()
        edge_pred = torch.from_numpy(edge_pred)
        proba_complproba = torch.stack([edge_pred, 1-edge_pred])
        proba_complproba = torch.softmax(proba_complproba, dim=0)
        max_inds = torch.argmax(proba_complproba, dim=0)
        edge_pred[max_inds == 1] = 1 - proba_complproba[1, max_inds == 1]
        edge_pred[max_inds == 0] = proba_complproba[0, max_inds == 0]

        node_labeling = run_watershed(gaussian_filter(edge_pred, sigma=.2), min_size=4)

def graphs_for_masked_data():
    for dir in [tgtdir_val]:
        fnames = sorted(glob(os.path.join(dir, 'raw_wtsd/*.h5')))
        pix_dir = os.path.join(dir, 'bg_masked_data/pix_data')
        graph_dir = os.path.join(dir, 'bg_masked_data/graph_data')
        for i in range(len(fnames)):
            fname = fnames[i]
            head, tail = os.path.split(fname)
            num = tail[6:-3]

            raw = h5py.File(fname, 'r')['raw'][:]
            gt = h5py.File(fname, 'r')['label'][:]

            affs = torch.from_numpy(h5py.File(os.path.join(dir, 'affinities_01_trainsz', tail[:-3] + '_predictions' + '.h5'), 'r')['predictions'][:]).squeeze(1)
            graph_file = h5py.File(os.path.join(graph_dir, "graph_" + num + ".h5"), 'a')
            pix_file = h5py.File(os.path.join(pix_dir, "pix_" + num + ".h5"), 'a')
            affs = torch.sigmoid(affs).numpy()
            #
            node_labeling = h5py.File(os.path.join(dir, "bg_masked_data/graph_" + num + ".h5"), 'r')["node_labeling"][:]
            #
            # # relabel to consecutive ints starting at 0
            node_labeling = torch.from_numpy(node_labeling.astype(np.long))
            gt = torch.from_numpy(gt.astype(np.long))
            mask = node_labeling[None] == torch.unique(node_labeling)[:, None, None]
            node_labeling = (mask * (torch.arange(len(torch.unique(node_labeling)), device=node_labeling.device)[:, None, None] + 1)).sum(0) - 1

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
            # edges = np.sort(edges, axis=-1)
            edges = edges.T


            pix_file.create_dataset("raw", data=raw, chunks=True)
            pix_file.create_dataset("gt", data=gt, chunks=True)
            #
            graph_file.create_dataset("edges", data=edges, chunks=True)
            graph_file.create_dataset("edge_feat", data=edge_feat, chunks=True)
            graph_file.create_dataset("diff_to_gt", data=diff_to_gt)
            graph_file.create_dataset("gt_edge_weights", data=gt_edge_weights, chunks=True)
            graph_file.create_dataset("node_labeling", data=node_labeling, chunks=True)
            graph_file.create_dataset("affinities", data=affs, chunks=True)
            graph_file.create_dataset("offsets", data=np.array([[1, 0], [0, 1], [2, 0], [0, 2]]), chunks=True)

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
    # graphs_for_masked_data()
    # check_sp()
    a=1
