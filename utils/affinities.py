import h5py
from affogato.affinities import compute_affinities
import numpy as np
import elf.segmentation.features as feats
import torch

def computeAffs(file_from, offsets):
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
    affinities = np.zeros([len(offsets)] + list(raw.shape[:2]))
    normed_raw = raw / raw.max()
    for y in range(normed_raw.shape[0]):
        for x in range(normed_raw.shape[1]):
            for i, off in enumerate(offsets):
                if 0 <= y+off[0] < normed_raw.shape[0] and 0 <= x+off[1] < normed_raw.shape[1]:
                    affinities[i, y, x, ...] = np.linalg.norm(normed_raw[y, x] - normed_raw[y+off[0], x+off[1]])
    return affinities

def get_affinities_from_embeddings_2d(embeddings, offsets, delta, p=2):
    """implementing eq. 6 in https://arxiv.org/pdf/1909.09872.pdf"""
    affs = torch.empty(((len(offsets), embeddings.shape[0]) + embeddings.shape[2:]), device=embeddings.device)
    for i, off in enumerate(offsets):
        rolled = torch.roll(embeddings, off, dims=(-2, -1))
        affs[i] = torch.maximum((delta - torch.norm(embeddings - rolled, p=p, dim=1)) / 2 * delta, torch.tensor([0], device=embeddings.device)) ** 2

    return affs

def get_stacked_node_data(nodes, edges, segmentation, raw, size):
    raw_nodes = torch.empty([len(nodes), *size])
    cms = torch.empty((len(nodes), 2))
    angles = torch.zeros(len(edges) * 2) - 11
    for i, n in enumerate(nodes):
        mask = (n == segmentation)
        # x, y = utils.bbox(mask.unsqueeze(0).numpy())
        # x, y = x[0], y[0]
        # masked_seg = mask.float() * raw
        # masked_seg = masked_seg[x[0]:x[1]+1, y[0]:y[1]+1]
        # if 0 in masked_seg.shape:
        #     a=1
        # raw_nodes[i] = torch.nn.functional.interpolate(masked_seg.unsqueeze(0).unsqueeze(0), size=size)
        idxs = torch.where(mask)
        cms[n.long()] = torch.tensor([torch.sum(idxs[0]).long(), torch.sum(idxs[1]).long()]) / mask.sum()
    for i, e in enumerate(edges):
        vec = cms[e[1]] - cms[e[0]]
        angle = abs(np.arctan(vec[0] / (vec[1] + np.finfo(float).eps)))
        if vec[0] <= 0 and vec[1] <= 0:
            angles[i] = np.pi + angle
            angles[i + len(edges)] = angle
        elif vec[0] >= 0 and vec[1] <= 0:
            angles[i] = np.pi - angle
            angles[i + len(edges)] = 2 * np.pi - angle
        elif vec[0] <= 0 and vec[1] >= 0:
            angles[i] = 2 * np.pi - angle
            angles[i + len(edges)] = np.pi - angle
        elif vec[0] >= 0 and vec[1] >= 0:
            angles[i] = angle
            angles[i + len(edges)] = np.pi + angle
        else:
            assert False
    if angles.max() > 2 * np.pi + 1e-20 or angles.min() + 1e-20 < 0:
        assert False
    angles = np.rint(angles / (2 * np.pi) * 63)
    return raw_nodes, angles.long()


def get_edge_features_1d(sp_seg, offsets, affinities):
    offsets_3d = []
    for off in offsets:
        offsets_3d.append([0] + off)

    rag = feats.compute_rag(np.expand_dims(sp_seg, axis=0))
    edge_feat = feats.compute_affinity_features(rag, np.expand_dims(affinities, axis=1), offsets_3d)[:, :]
    return edge_feat, rag.uvIds()

