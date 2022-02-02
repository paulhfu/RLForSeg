import os
import torch
from glob import glob
import h5py
import numpy as np
from utils.affinities import get_naive_affinities, get_edge_features_1d
from utils.general import calculate_gt_edge_costs, set_seed_everywhere, get_contour_from_2d_binary
from elf.segmentation.watershed import watershed, apply_size_filter

set_seed_everywhere(19)

def store_all(base_dir, n_samples, fnames):
    import threading
    thds=[]

    for i in range(n_samples // 3):
        thds.append(threading.Thread(target=store_, args=(base_dir, fnames["images"][i],
                        fnames["masks"][i],
                        fnames["boundaries"][i],
                        fnames["semantic"][i],
                        fnames["superpixels"][i])))
        thds[-1].start()
        thds[-1].join()

    for i in range(n_samples // 3, 2 * (n_samples // 3)):
        thds.append(threading.Thread(target=store_, args=(base_dir, fnames["images"][i],
                        fnames["masks"][i],
                        fnames["boundaries"][i],
                        fnames["semantic"][i],
                        fnames["superpixels"][i])))
        thds[-1].start()
        thds[-1].join()

    for i in range(2 * (n_samples // 3), n_samples):
        thds.append(threading.Thread(target=store_, args=(base_dir, fnames["images"][i],
                        fnames["masks"][i],
                        fnames["boundaries"][i],
                        fnames["semantic"][i],
                        fnames["superpixels"][i])))
        thds[-1].start()
        thds[-1].join()

def store_(base_dir, imname, maskname, bndname, semanticname, supixname):
    dev = "cuda:0"
    from skimage import io

    raw = io.imread(imname)[..., np.newaxis].astype(np.float32)
    gt = torch.from_numpy(io.imread(maskname)[..., np.newaxis].astype(np.int64)).to(dev).squeeze()
    bnd = io.imread(bndname)[..., np.newaxis].astype(np.float32)
    semantic = io.imread(semanticname)[..., np.newaxis].astype(np.float32)
    node_labeling = h5py.File(supixname, "r")['gt_intersected'][:].astype(np.uint32)
    node_labeling, _ = apply_size_filter(node_labeling, semantic.squeeze(), 3)


    onehot = gt[None] == torch.unique(gt)[:, None, None]
    masses = (onehot * (torch.arange(onehot.shape[0], device=onehot.device)[:, None, None] + 1)).flatten(1).sum(1)
    print(masses.min().item())
    print(masses.max().item())

    node_labeling = torch.from_numpy(node_labeling.astype(np.long)).to(dev)
    onehot = node_labeling[None] == torch.unique(node_labeling)[:, None, None]
    node_labeling = (onehot * (torch.arange(onehot.shape[0], device=onehot.device)[:, None, None] + 1)).sum(0)

    raw -= raw.min()
    raw /= raw.max()
    bnd -= bnd.min()
    bnd /= bnd.max()
    semantic -= semantic.min()
    semantic /= semantic.max()
    gt, edges, edge_feat, node_feat, gt_edge_weights, node_labeling, nodes, affinities, offsets = get_graphs(node_labeling, bnd, gt, raw, dev)
    #continue
    # img = np.concatenate([raw.squeeze()[np.newaxis], bnd.squeeze()[np.newaxis]], 0)
    img = raw.squeeze()[np.newaxis]

    file = h5py.File(os.path.join(base_dir, os.path.basename(imname[:-4]) + ".h5"), 'w')
    file.create_dataset("raw_2chnl", data=img, chunks=True)
    file.create_dataset("raw", data=raw.squeeze()[np.newaxis], chunks=True)
    file.create_dataset("gt", data=gt, chunks=True)
    file.create_dataset("edges", data=edges, chunks=True)
    file.create_dataset("edge_feat", data=edge_feat, chunks=True)
    file.create_dataset("node_feat", data=node_feat, chunks=True)
    file.create_dataset("gt_edge_weights", data=gt_edge_weights, chunks=True)
    file.create_dataset("node_labeling", data=node_labeling, chunks=True)
    file.create_dataset("affinities", data=affinities, chunks=True)
    file.create_dataset("offsets", data=np.array(offsets), chunks=True)
    file.close()

def get_graphs(node_labeling, bnd, gt, img, dev="cuda:0"):
    edge_offsets = [[-1, 0], [0, -1]]
    node_labeling -= node_labeling.min()

    nodes = torch.unique(node_labeling)
    onehot = torch.zeros((int(len(nodes)),) + node_labeling.size(), device=dev, dtype=torch.long).scatter_(0, node_labeling[None], 1)
    #node_labeling = (torch.arange(onehot.shape[0], device=dev)[:, None, None] * onehot).sum(0)
    masses = onehot.flatten(1).sum(-1)
    node_feat = torch.sigmoid((20 + masses)/500).cpu().numpy()[np.newaxis]

    nodes = nodes.cpu().numpy()
    node_labeling = node_labeling.cpu().numpy()
    affinities = get_naive_affinities(bnd, edge_offsets)
    try:
        assert all(nodes == np.array(range(len(nodes)), dtype=np.float))
    except:
        Warning("node ids are off")

    # get edges from node labeling and edge features from affinity stats
    edge_feat, neighbors = get_edge_features_1d(node_labeling, edge_offsets, affinities)
    # get gt edge weights based on edges and gt image
    gt_edge_weights = calculate_gt_edge_costs(torch.from_numpy(neighbors.astype(np.int64)).to(dev),
                                              torch.from_numpy(node_labeling.squeeze()).to(dev),
                                                gt, 0.5)
    gt = gt.cpu().numpy()
    gt_edge_weights = gt_edge_weights.cpu().numpy()
    edges = neighbors.astype(np.long)

    affinities = affinities.astype(np.float32)
    edge_feat = edge_feat.astype(np.float32).transpose(1,0)
    nodes = nodes.astype(np.float32)
    node_labeling = node_labeling.astype(np.float32)
    gt_edge_weights = gt_edge_weights.astype(np.float32)

    edges = np.sort(edges, axis=-1)
    edges = edges.T

    return gt, edges, edge_feat, node_feat,  gt_edge_weights, node_labeling, nodes, affinities, edge_offsets


if __name__ == "__main__":
    import json
    filterstring = "nuclei-medium"
    dir = "/g/kreshuk/pape/Work/data/data_science_bowl/dsb2018"
    test_images = {}
    train_images = {}
    dev = "cuda:0"

    with open('/g/kreshuk/pape/Work/data/data_science_bowl/dsb2018/annotations.json') as json_file:
        id_dict = json.load(json_file)
        for chnl in ["boundaries", "superpixels", "semantic", "images", "masks"]:
            images = sorted(glob(os.path.join(dir, "test", chnl, '*.tif')))
            if chnl == "boundaries":
                images = [image for image in images if image[-5] == "1"]
                test_images[chnl] = [image for image in images if id_dict[os.path.basename(image)[:-7]] == filterstring]
            elif chnl == "semantic":
                images = sorted(glob(os.path.join(dir, "test", "boundaries", '*.tif')))
                images = [image for image in images if image[-5] == "0"]
                test_images[chnl] = [image for image in images if id_dict[os.path.basename(image)[:-7]] == filterstring]
            elif chnl == "superpixels":
                images = sorted(glob(os.path.join(dir, "test", chnl, '*.h5')))
                test_images[chnl] = [image for image in images if id_dict[os.path.basename(image)[:-3]] == filterstring]
            else:
                test_images[chnl] = [image for image in images if id_dict[os.path.basename(image)[:-4]] == filterstring]

        for chnl in ["boundaries", "superpixels", "semantic", "images", "masks"]:
            images = sorted(glob(os.path.join(dir, "train", chnl, '*.tif')))
            if chnl == "boundaries":
                images = [image for image in images if image[-5] == "1"]
                train_images[chnl] = [image for image in images if id_dict[os.path.basename(image)[:-7]] == filterstring]
            elif chnl == "semantic":
                images = sorted(glob(os.path.join(dir, "train", "boundaries", '*.tif')))
                images = [image for image in images if image[-5] == "0"]
                train_images[chnl] = [image for image in images if id_dict[os.path.basename(image)[:-7]] == filterstring]
            elif chnl == "superpixels":
                images = sorted(glob(os.path.join(dir, "train", chnl, '*.h5')))
                train_images[chnl] = [image for image in images if id_dict[os.path.basename(image)[:-3]] == filterstring]
            else:
                train_images[chnl] = [image for image in images if id_dict[os.path.basename(image)[:-4]] == filterstring]

    store_all("/g/kreshuk/hilt/projects/data/dsb/medium/test", len(test_images["masks"]), test_images)
    store_all("/g/kreshuk/hilt/projects/data/dsb/medium/train", len(train_images["masks"]), train_images)