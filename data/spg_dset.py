import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import h5py
import os
from glob import glob
from skimage import draw
from utils.patch_manager import StridedRollingPatches2D, StridedPatches2D, NoPatches2D
from utils.graphs import squeeze_repr
import torch.utils.data as torch_data
import numpy as np
import warnings
# import torchvision


class SpgDset(torch_data.Dataset):
    def __init__(self, root_dir, patch_mngr, keys, n_edges_min=0):
        """ dataset for loading images (raw, gt, superpixel segs) and according rags"""
        # self.transform = torchvision.transforms.Normalize(0, 1, inplace=False)
        self.keys = keys
        self.graph_dir = os.path.join(root_dir, 'graph_data')
        self.pix_dir = os.path.join(root_dir, 'pix_data')
        self.graph_file_names = sorted(glob(os.path.join(self.graph_dir, "*.h5")))
        self.pix_file_names = sorted(glob(os.path.join(self.pix_dir, "*.h5")))
        self.reorder_sp = patch_mngr.reorder_sp
        self.n_edges_min = n_edges_min
        pix_file = h5py.File(self.pix_file_names[0], 'r')
        shape = pix_file[keys.raw][:].shape[-2:]
        if patch_mngr.name == "rotated":
            self.pm = StridedRollingPatches2D(patch_mngr.patch_stride, patch_mngr.patch_shape, shape)
        elif patch_mngr.name == "no_cross":
            self.pm = StridedPatches2D(patch_mngr.patch_stride, patch_mngr.patch_shape, shape)
        else:
            self.pm = NoPatches2D()
        self.length = len(self.graph_file_names) * np.prod(self.pm.n_patch_per_dim)

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
        a2.imshow(cm.prism(gt / gt.max()))
        a2.set_title('gt')
        a3.imshow(cm.prism(sp_seg / sp_seg.max()))
        a3.set_title('sp')
        plt.tight_layout()
        plt.show()

    def __getitem__(self, idx):
        img_idx = idx // np.prod(self.pm.n_patch_per_dim)
        patch_idx = idx % np.prod(self.pm.n_patch_per_dim)
        pix_file = h5py.File(self.pix_file_names[img_idx], 'r')
        graph_file = h5py.File(self.graph_file_names[img_idx], 'r')

        raw = pix_file[self.keys.raw][:].squeeze()
        if raw.ndim == 2:
            raw = torch.from_numpy(raw.astype(np.float)).float()[None]
        elif raw.shape[-1] == 3:
            raw = torch.from_numpy(raw.astype(np.float)).permute(2, 0, 1).float()
        else:
            raw = torch.from_numpy(raw.astype(np.float)).float()

        if "raw_1" in self.keys:
            raw_1 = torch.from_numpy(pix_file[self.keys.raw_1][:].squeeze().astype(np.float)).float()[None]
            raw = torch.cat((raw, raw_1), 0)

        # raw -= raw.min()
        # raw /= raw.max()
        # def _draw(a, b, rs, idx):
        #     rw = h5py.File(self.pix_file_names[idx], 'r')[self.keys.raw][:]
        #     altered_raw = np.stack([rw, rw, rw], -1)
        #     for r in rs:
        #         raw_circle = draw.circle_perimeter(a, b, r, method='bresenham', shape=raw.shape[-2:])
        #         altered_raw[raw_circle[0], raw_circle[1], 0] = 1
        #     return altered_raw
        # plt.imshow(_draw(400, 350, [210, 280, 320], 0));plt.show()


        if self.keys.node_labeling in pix_file.keys():
            sp_seg = torch.from_numpy(pix_file[self.keys.node_labeling][:].astype(np.float))
        else:
            sp_seg = torch.from_numpy(graph_file[self.keys.node_labeling][:].astype(np.float))
        if "gt" in self.keys:
            gt = torch.from_numpy(pix_file[self.keys.gt][:].astype(np.float))
        else:
            gt = torch.zeros_like(sp_seg)

        all = torch.cat([raw, gt[None], sp_seg[None]], 0)
        patch = self.pm.get_patch(all, patch_idx)

        sp_seg = patch[-1]
        gt = patch[-2]
        raw = patch[:-2].float()
        # raw = torch.stack((raw[0], raw[0]), 0)
        # if torch.randint(0, 10, (1, )) > 6:
        #     raw += torch.randn_like(raw) * 0.1
        #     raw[0] -= raw[0].min()
        #     raw[0] /= raw[0].max()
        #     raw[1] -= raw[1].min()
        #     raw[1] /= raw[1].max()

        if not self.reorder_sp:
            return raw, gt.long(), sp_seg.long(), torch.tensor([img_idx])

        # relabel to consecutive ints starting at 0
        mask = sp_seg[None] == torch.unique(sp_seg)[:, None, None]
        sp_seg = (mask * (torch.arange(len(torch.unique(sp_seg)), device=sp_seg.device)[:, None, None] + 1)).sum(0) - 1

        mask = gt[None] == torch.unique(gt)[:, None, None]
        gt = (mask * (torch.arange(len(torch.unique(gt)), device=gt.device)[:, None, None] + 1)).sum(0) - 1

        return raw, gt.long()[None], sp_seg.long()[None], torch.tensor([img_idx])

    def get_graphs(self, indices, patches, device="cpu"):
        edges, edge_feat,  gt_edge_weights = [], [], []
        for i, patch in zip(indices, patches):
            nodes = torch.unique(patch).unsqueeze(-1).unsqueeze(-1)
            try:
                graph_file = h5py.File(self.graph_file_names[i], 'r')
            except Exception as e:
                warnings.warn("could not find dataset")

            es = torch.from_numpy(graph_file[self.keys.edges][:]).to(device)
            iters = (es.unsqueeze(0) == nodes).float().sum(0).sum(0) >= 2
            es = es[:, iters]
            squeeze_repr(nodes.squeeze(-1).squeeze(-1), es, patch.squeeze(0))

            edges.append(es)
            edge_feat.append(torch.from_numpy(graph_file[self.keys.edge_feat][:]).to(device)[iters])
            if "gt_edge_weights" in self.keys:
                gt_edge_weights.append(torch.from_numpy(graph_file[self.keys.gt_edge_weights][:]).to(device)[iters])
            else:
                gt_edge_weights = None

            if es.shape[1] < self.n_edges_min:
                return None
        return edges, edge_feat, gt_edge_weights


if __name__ == "__main__":
    set = SpgDset()
    ret = set.get(3)
    a = 1
