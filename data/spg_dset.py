import torch
import h5py
import os
from glob import glob
from utils.patch_manager import StridedRollingPatches2D, StridedPatches2D, NoPatches2D
from utils.graphs import squeeze_repr
import torch.utils.data as torch_data
import numpy as np


class SpgDset(torch_data.Dataset):
    def __init__(self, file_dir, patch_mngr, keys, n_edges_min=0):
        """ dataset for loading images (raw, gt, superpixel segs) and according rags"""
        # self.transform = torchvision.transforms.Normalize(0, 1, inplace=False)
        self.keys = keys
        self.file_names = sorted(glob(os.path.join(file_dir, "*.h5")))
        self.reorder_sp = patch_mngr.reorder_sp
        self.n_edges_min = n_edges_min
        example_file = h5py.File(self.file_names[0], 'r')
        shape = example_file[keys.raw][:].shape[-2:]
        if patch_mngr.name == "rotated":
            self.pm = StridedRollingPatches2D(patch_mngr.patch_stride, patch_mngr.patch_shape, shape)
        elif patch_mngr.name == "no_cross":
            self.pm = StridedPatches2D(patch_mngr.patch_stride, patch_mngr.patch_shape, shape)
        else:
            self.pm = NoPatches2D()
        self.length = len(self.file_names) * np.prod(self.pm.n_patch_per_dim)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_idx = idx // np.prod(self.pm.n_patch_per_dim)
        patch_idx = idx % np.prod(self.pm.n_patch_per_dim)
        file = h5py.File(self.file_names[img_idx], 'r')

        raw = torch.from_numpy(file[self.keys.raw][:])
        sp_seg = torch.from_numpy(file[self.keys.superpixels][:].astype(np.float))
        if "gt" in self.keys:  # in case we have ground truth
            gt = torch.from_numpy(file[self.keys.gt][:].astype(np.float))
        else:
            gt = torch.zeros_like(sp_seg)

        all = torch.cat([raw, gt[None], sp_seg[None]], 0)
        patch = self.pm.get_patch(all, patch_idx)

        sp_seg = patch[-1]
        gt = patch[-2]
        raw = patch[:-2].float()

        if not self.reorder_sp:
            return raw, gt.long(), sp_seg.long(), torch.tensor([img_idx])

        # relabel to consecutive ints starting at 0
        mask = sp_seg[None] == torch.unique(sp_seg)[:, None, None]
        sp_seg = (mask * (torch.arange(len(torch.unique(sp_seg)), device=sp_seg.device)[:, None, None] + 1)).sum(0) - 1

        mask = gt[None] == torch.unique(gt)[:, None, None]
        gt = (mask * (torch.arange(len(torch.unique(gt)), device=gt.device)[:, None, None] + 1)).sum(0) - 1

        return raw, gt.long()[None], sp_seg.long()[None], torch.tensor([img_idx])

    def get_graphs(self, indices, patches, device="cpu"):
        # we get the graph data separately because it cannot be batched
        edges,  gt_edge_weights, edge_feat, node_feat = [], [], [], []
        for i, patch in zip(indices, patches):
            nodes = torch.unique(patch).unsqueeze(-1).unsqueeze(-1)
            file = h5py.File(self.file_names[i], 'r')
            # get only the part of the graph that is visible in the patch
            es = torch.from_numpy(file[self.keys.edges][:]).to(device)
            iters = (es.unsqueeze(0) == nodes).float().sum(0).sum(0) >= 2
            es = es[:, iters]
            squeeze_repr(nodes.squeeze(-1).squeeze(-1), es, patch.squeeze(0))

            edges.append(es)
            if "gt_edge_weights" in self.keys:
                gt_edge_weights.append(torch.from_numpy(file[self.keys.gt_edge_weights][:]).to(device)[iters])
            else:
                gt_edge_weights = None
            if "edge_feat" in self.keys:
                edge_feat.append(torch.from_numpy(file[self.keys.edge_feat][:]).to(device)[:, iters].permute((1, 0)))
            else:
                edge_feat = None
            if "node_feat" in self.keys:
                node_feat.append(torch.from_numpy(file[self.keys.node_feat][:]).to(device)[:, nodes.squeeze()].permute((1, 0)))
            else:
                node_feat = None

            assert es.shape[1] >= self.n_edges_min, "One of the graphs was smaller than our min size given by largest subgraph size"
        return edges, gt_edge_weights, edge_feat, node_feat


if __name__=="__main__":
    from utils.graphs import get_position_mass_in_rag
    dir = "/g/kreshuk/hilt/projects/data/color_circles/train"
    newdir = "/g/kreshuk/hilt/projects/data/color_circles/all_in_one/train"
    graph_file_names = sorted(glob(os.path.join(dir + "/graph_data", "*.h5")))

    raw_n = "raw_4chnl"
    gt_n = "gt"
    node_labeling_n = "node_labeling"
    gt_edge_weights_n = "gt_edge_weights"
    edges_n = "edges"

    for gfn in graph_file_names:
        _, name = os.path.split(gfn)
        name = name[6:]
        pfn = os.path.join(dir, "pix_data", "pix_"+name)
        pf = h5py.File(pfn, "r")
        gf = h5py.File(gfn, "r")

        newfile = h5py.File(os.path.join(newdir, name), "r+")

        edge_features = torch.from_numpy(gf["edge_feat"][:])
        # gt = pf[gt_n][:]
        sp_seg = gf[node_labeling_n][:]
        # gt_edges = gf[gt_edge_weights_n][:]
        edges = gf[edges_n][:]

        edge_angles, sp_feats = get_position_mass_in_rag(torch.from_numpy(edges), torch.from_numpy(sp_seg))

        # newfile.create_dataset(name="raw", data=raw)
        # newfile.create_dataset(name="superpixels", data=sp_seg)
        # newfile.create_dataset(name="gt", data=gt)
        # newfile.create_dataset(name="edges", data=edges)
        # newfile.create_dataset(name="gt_edge_weights", data=gt_edges)

        edge_features = torch.cat([edge_angles[:, None], edge_features[:, :2]], 1).float().permute((1, 0))

        newfile.create_dataset(name="edge_feat", data=edge_features)
        newfile.create_dataset(name="node_feat", data=sp_feats)

        newfile.close()
        pf.close()
        gf.close()

