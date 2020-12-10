import h5py
from affogato.affinities import compute_affinities
import numpy as np
import elf.segmentation.features as feats
import torch

def computeAffs(file_from, offsets):
    """computes affinities of a segmentation"""
    file = h5py.File(file_from, 'a')
    keys = list(file.keys())
    file.create_group('masks')
    file.create_group('affs')
    for k in keys:
        data = file[k][:].copy()
        affinities, _ = compute_affinities(data != 0, offsets)
        file['affs'].create_dataset(k, data=affinities)
        file['masks'].create_dataset(k, data=data)
        del file[k]
    return

def get_naive_affinities(raw, offsets):
    """get naive pixel affinities based on differences in pixel intensities."""
    affinities = []
    for i, off in enumerate(offsets):
        rolled = np.roll(raw, tuple(-np.array(off)), (0, 1))
        dist = np.linalg.norm(raw - rolled, axis=-1)
        affinities.append(dist / dist.max())
    return np.stack(affinities)

def get_affinities_from_embeddings_2d(embeddings, offsets, delta, p=2):
    """implementing eq. 6 in https://arxiv.org/pdf/1909.09872.pdf"""
    affs = torch.empty(((len(offsets), embeddings.shape[0]) + embeddings.shape[2:]), device=embeddings.device)
    for i, off in enumerate(offsets):
        rolled = torch.roll(embeddings, tuple(-np.array(off)), dims=(-2, -1))
        affs[i] = torch.maximum((delta - torch.norm(embeddings - rolled, p=p, dim=1)) / 2 * delta, torch.tensor([0], device=embeddings.device)) ** 2

    return affs

def get_edge_features_1d(sp_seg, offsets, affinities):
    offsets_3d = []
    for off in offsets:
        offsets_3d.append([0] + off)

    rag = feats.compute_rag(np.expand_dims(sp_seg, axis=0))
    edge_feat = feats.compute_affinity_features(rag, np.expand_dims(affinities, axis=1), offsets_3d)[:, :]
    return edge_feat, rag.uvIds()

