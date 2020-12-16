import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import h5py
import os
from glob import glob
from utils.patch_manager import StridedRollingPatches2D, StridedPatches2D, NoPatches2D
from utils.graphs import squeeze_repr
import torch.utils.data as torch_data
import numpy as np
import warnings
import torchvision


class SpgDset(torch_data.Dataset):
    def __init__(self, root_dir, n_edges_min=0, patch_manager="", patch_stride=None, patch_shape=None, reorder_sp=False):
        """ dataset for loading images (raw, gt, superpixel segs) and according rags"""
        self.transform = torchvision.transforms.Normalize(0, 1, inplace=False)
        self.graph_dir = os.path.join(root_dir, 'graph_data')
        self.pix_dir = os.path.join(root_dir, 'pix_data')
        self.graph_file_names = sorted(glob(os.path.join(self.graph_dir, "*.h5")))
        self.pix_file_names = sorted(glob(os.path.join(self.pix_dir, "*.h5")))
        self.reorder_sp = reorder_sp
        self.n_edges_min = n_edges_min
        pix_file = h5py.File(self.pix_file_names[0], 'r')
        shape = pix_file["gt"][:].shape
        if patch_manager == "rotated":
            self.pm = StridedRollingPatches2D(patch_stride, patch_shape, shape)
        elif patch_manager == "no_cross":
            self.pm = StridedPatches2D(patch_stride, patch_shape, shape)
        else:
            self.pm = NoPatches2D()
        self.length = len(self.graph_file_names) * np.prod(self.pm.n_patch_per_dim)
        print('found ', self.length, " data patches")

    def __len__(self):
        return self.length

    def viewItem(self, idx):
        pix_file = h5py.File(self.pix_file_names[idx], 'r')
        graph_file = h5py.File(self.graph_file_names[idx], 'r')

        raw = pix_file["raw"][:]
        gt = pix_file["gt"][:]
        sp_seg = graph_file["node_labeling"][:]

        fig, (a1, a2, a3) = plt.subplots(1, 3, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
        a1.imshow(raw, cmap='gray')
        a1.set_title('raw')
        a2.imshow(cm.prism(gt/gt.max()))
        a2.set_title('gt')
        a3.imshow(cm.prism(sp_seg/sp_seg.max()))
        a3.set_title('sp')
        plt.tight_layout()
        plt.show()



    def __getitem__(self, idx):
        img_idx = idx // np.prod(self.pm.n_patch_per_dim)
        patch_idx = idx % np.prod(self.pm.n_patch_per_dim)
        pix_file = h5py.File(self.pix_file_names[img_idx], 'r')
        graph_file = h5py.File(self.graph_file_names[img_idx], 'r')

        raw = pix_file["raw"][:]
        if raw.ndim == 2:
            raw = torch.from_numpy(raw).float().unsqueeze(0)
        else:
            raw = torch.from_numpy(raw).permute(2, 0, 1).float()
        gt = torch.from_numpy(pix_file["gt"][:]).unsqueeze(0).float()
        sp_seg = torch.from_numpy(graph_file["node_labeling"][:]).unsqueeze(0).float()

        all = torch.cat([raw, gt, sp_seg], 0)
        patch = self.pm.get_patch(all, patch_idx)

        sp_seg = patch[-1].unsqueeze(0)
        gt = patch[-2].unsqueeze(0)
        new_sp_seg = torch.zeros_like(sp_seg)
        new_gt = torch.zeros_like(gt)

        if not self.reorder_sp:
            return patch[:-2], gt, sp_seg, torch.tensor([img_idx])

        un = torch.unique(sp_seg)
        for i, sp in enumerate(un):
            new_sp_seg[sp_seg == sp] = i
        for i, obj in enumerate(torch.unique(gt)):
            new_gt[gt == obj] = i

        return self.transform(patch[:-2]), new_gt, new_sp_seg, torch.tensor([img_idx])


    def get_graphs(self, indices, patches, device="cpu"):
        edges, edge_feat, diff_to_gt, gt_edge_weights = [], [], [], []
        for i, patch in zip(indices, patches):
            nodes = torch.unique(patch).unsqueeze(-1).unsqueeze(-1)
            try:
                graph_file = h5py.File(self.graph_file_names[i], 'r')
            except Exception as e:
                warnings.warn("could not find dataset")

            es = torch.from_numpy(graph_file["edges"][:]).to(device)
            iters = (es.unsqueeze(0) == nodes).float().sum(0).sum(0) >= 2
            es = es[:, iters]
            squeeze_repr(nodes.squeeze(-1).squeeze(-1), es, patch.squeeze(0))

            edges.append(es)
            edge_feat.append(torch.from_numpy(graph_file["edge_feat"][:]).to(device)[iters])
            diff_to_gt.append(torch.tensor(graph_file["diff_to_gt"][()], device=device))
            gt_edge_weights.append(torch.from_numpy(graph_file["gt_edge_weights"][:]).to(device)[iters])


            if es.shape[1] < self.n_edges_min:
                return None
        return edges, edge_feat, diff_to_gt, gt_edge_weights

if __name__ == "__main__":
    set = SpgDset()
    ret = set.get(3)
    a=1