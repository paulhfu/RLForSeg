import numpy as np
import torch
import collections
import elf
import wandb
import matplotlib.pyplot as plt
from elf.segmentation.multicut import multicut_kernighan_lin
from elf.segmentation.features import project_node_labels_to_pixels
from rag_utils import find_dense_subgraphs

import rewards
from utils.graphs import collate_edges, get_edge_indices
from utils.general import random_label_cmap

State = collections.namedtuple("State", ["raw", "sp_seg", "edge_ids", "edge_feat", "node_feat", "subgraph_indices",
                                         "sep_subgraphs",  "gt_edge_weights"])


class MulticutEmbeddingsEnv:

    def __init__(self, cfg, device):
        super(MulticutEmbeddingsEnv, self).__init__()

        self.reset()
        self.cfg = cfg
        self.device = device
        self.max_p = torch.nn.MaxPool2d(3, padding=1, stride=1)
        self.reward_function = eval("rewards." + self.cfg.reward_function)(cfg.s_subgraph)


    def execute_action(self, actions, logg_vals=None, post_stats=False, post_images=False, tau=None, train=True):
        self.current_edge_weights = actions.squeeze()
        self.current_soln = self.get_current_soln(self.current_edge_weights)

        self.sg_current_edge_weights = []
        for i, sz in enumerate(self.cfg.s_subgraph):
            self.sg_current_edge_weights.append(
                self.current_edge_weights[self.subgraph_indices[i].view(-1, sz)])

            reward = self.reward_function(prediction_segmentation=self.current_soln.long(),
                                                                      gt=self.sg_gt_edges, dir_edges=self.dir_edge_ids,
                                                                      superpixel_segmentation=self.init_sp_seg.long(),
                                                                      actions=actions,
                                                                      subgraph_indices=self.subgraph_indices)

        if post_stats:
            tag = "train/" if train else "validation/"
            wandb.log({tag + "avg_return": reward[-1].item()})
            if post_images:
                mc_soln = self.gt_soln[-1].cpu() if self.gt_edge_weights is not None else torch.zeros(self.raw.shape[-2:])
                wandb.log({tag + "pred_mean": wandb.Histogram(self.current_edge_weights.view(-1).cpu().numpy())})
                fig, axes = plt.subplots(2, 3, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
                axes[0, 0].imshow(self.gt_seg[-1].cpu().squeeze(), cmap=random_label_cmap(), interpolation="none")
                axes[0, 0].set_title('gt')
                if self.raw.ndim == 3:
                    axes[0, 1].imshow(self.raw[-1, 0])
                else:
                    axes[0, 1].imshow(self.raw[-1])
                axes[0, 1].set_title('raw image')
                axes[0, 2].imshow(self.init_sp_seg[-1].cpu(), cmap=random_label_cmap(), interpolation="none")
                axes[0, 2].set_title('superpixels')
                axes[1, 0].imshow(mc_soln, cmap=random_label_cmap(), interpolation="none")
                axes[1, 0].set_title('mc_gt')
                axes[1, 1].imshow(self.init_sp_seg[-1].cpu())
                axes[1, 1].set_title('embed')
                axes[1, 2].imshow(self.current_soln[-1].cpu(), cmap=random_label_cmap(), interpolation="none")
                axes[1, 2].set_title('prediction')
                wandb.log({tag: [wandb.Image(fig, caption="state")]})
                plt.close('all')
            if logg_vals is not None:
                for key, val in logg_vals.items():
                    wandb.log({tag + key: val})

        self.acc_reward.append(reward[-1].item())
        return reward

    def get_state(self):
        return State(self.raw, self.batched_sp_seg, self.edge_ids, self.edge_feat, self.node_feat,
                     self.subgraph_indices, self.sep_subgraphs, self.gt_edge_weights)

    def update_data(self, raw, gt, edge_ids, gt_edges, sp_seg, rags, edge_feat, node_feat, *args, **kwargs):
        bs = raw.shape[0]
        dev = raw.device
        for _sp_seg in sp_seg:
            assert all(_sp_seg.unique() == torch.arange(_sp_seg.max() + 1, device=dev))

        self.rags = rags
        self.gt_seg, self.init_sp_seg = gt.squeeze(1), sp_seg.squeeze(1)
        self.raw = raw
        self.node_feat = node_feat if node_feat is None else torch.cat(node_feat, 0)
        self.edge_feat = edge_feat if edge_feat is None else torch.cat(edge_feat, 0)

        subgraphs, self.sep_subgraphs = [], []
        _subgraphs, _sep_subgraphs = find_dense_subgraphs([eids.transpose(0, 1).cpu().numpy() for eids in edge_ids], self.cfg.s_subgraph)
        _subgraphs = [torch.from_numpy(sg.astype(np.int64)).to(dev).permute(2, 0, 1) for sg in _subgraphs]
        _sep_subgraphs = [torch.from_numpy(sg.astype(np.int64)).to(dev).permute(2, 0, 1) for sg in _sep_subgraphs]

        self.dir_edge_ids = [torch.cat([_edge_ids, torch.stack([_edge_ids[1], _edge_ids[0]], dim=0)], dim=1) for _edge_ids in edge_ids]
        self.n_nodes = [eids.max() + 1 for eids in edge_ids]
        self.edge_ids, (self.n_offs, self.e_offs) = collate_edges(edge_ids)
        for i in range(len(self.cfg.s_subgraph)):
            subgraphs.append(torch.cat([sg + self.n_offs[i] for i, sg in enumerate(_subgraphs[i*bs:(i+1)*bs])], -2).flatten(-2, -1))
            self.sep_subgraphs.append(torch.cat(_sep_subgraphs[i*bs:(i+1)*bs], -2).flatten(-2, -1))

        self.subgraphs = subgraphs
        self.subgraph_indices = get_edge_indices(self.edge_ids, subgraphs)

        batched_sp = []
        for sp, off in zip(self.init_sp_seg, self.n_offs):
            batched_sp.append(sp + off)
        self.batched_sp_seg = torch.stack(batched_sp, 0)

        self.gt_edge_weights = gt_edges
        if self.gt_edge_weights is not None:
            self.gt_edge_weights = torch.cat(self.gt_edge_weights)
            self.gt_soln = self.get_current_soln(self.gt_edge_weights)
            self.sg_gt_edges = [self.gt_edge_weights[sg].view(-1, sz) for sz, sg in
                                zip(self.cfg.s_subgraph, self.subgraph_indices)]

        self.current_edge_weights = torch.ones(self.edge_ids.shape[1], device=self.edge_ids.device) / 2

        return

    def get_current_soln(self, edge_weights):
        p_min = 0.001
        p_max = 1.
        segmentations = []
        for i in range(1, len(self.e_offs)):
            probs = edge_weights[self.e_offs[i-1]:self.e_offs[i]]
            probs -= probs.min()
            probs /= probs.max()
            costs = (p_max - p_min) * probs + p_min
            costs = (torch.log((1. - costs) / costs)).detach().cpu().numpy()
            node_labels = elf.segmentation.multicut.multicut_decomposition(self.rags[i-1], costs, internal_solver='greedy-additive', n_threads=4)
            mc_seg = project_node_labels_to_pixels(self.rags[i-1], node_labels).squeeze()

            mc_seg = torch.from_numpy(mc_seg.astype(np.long)).to(self.device)
            # mask = mc_seg[None] == torch.unique(mc_seg)[:, None, None]
            # mc_seg = (mask * (torch.arange(len(torch.unique(mc_seg)), device=mc_seg.device)[:, None, None] + 1)).sum(0) - 1

            segmentations.append(mc_seg)
        return torch.stack(segmentations, dim=0)

    def reset(self):
        self.acc_reward = []
