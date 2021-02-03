from matplotlib import cm
import numpy as np
import torch
import collections
import matplotlib.pyplot as plt
from utils.reward_functions import UnSupervisedReward, SubGraphDiceReward
from rewards.artificial_cells_reward import ArtificialCellsReward
from rewards.leptin_data_reward_2d import LeptinDataReward2D
from utils.graphs import collate_edges, get_edge_indices, get_angles_smass_in_rag
from utils.general import get_angles, pca_project
from rag_utils import find_dense_subgraphs
import elf
from elf.segmentation.multicut import multicut_kernighan_lin
import nifty
from affogato.segmentation import compute_mws_segmentation
import h5py
from glob import glob

class MulticutEmbeddingsEnv():

    State = collections.namedtuple("State", ["node_embeddings", "edge_ids", "edge_angles", "sup_masses", "subgraph_indices", "sep_subgraphs", "round_n", "gt_edge_weights"])
    def __init__(self, embedding_net, cfg, device, writer=None, writer_counter=None):
        super(MulticutEmbeddingsEnv, self).__init__()

        self.embedding_net = embedding_net
        self.reset()
        self.cfg = cfg
        self.device = device
        self.writer = writer
        self.writer_counter = writer_counter
        self.last_final_reward = torch.tensor([0.0])
        self.max_p = torch.nn.MaxPool2d(3, padding=1, stride=1)
        self.aff_offsets = [[0, -1], [-1, 0], [-1, -1], [-8, 0], [0, -8]]
        self.sep_chnl = 3

        if self.cfg.sac.reward_function == 'sub_graph_dice':
            self.reward_function = SubGraphDiceReward()
        elif self.cfg.sac.reward_function == 'artificial_cells':
            fnames_pix = sorted(glob('/g/kreshuk/kaziakhm/circles_s025_gs0035_ps04_alln/pix_data/*.h5'))
            gts = [torch.from_numpy(h5py.File(fnames_pix[7], 'r')['gt'][:]).to(device)]
            gts.append(torch.from_numpy(h5py.File(fnames_pix[3], 'r')['gt'][:]).to(device))
            sample_shapes = []
            for gt in gts:
                # set gt to consecutive integer labels
                _gt = torch.zeros_like(gt).long()
                for _lbl, lbl in enumerate(torch.unique(gt)):
                    _gt += (gt == lbl).long() * _lbl
                gt = _gt
                sample_shapes.append(torch.zeros((int(gt.max()) + 1,) + gt.size(), device=device).scatter_(0, gt[None], 1)[
                                1:])  # 0 should be background

            self.reward_function = ArtificialCellsReward(torch.cat(sample_shapes))
        elif self.cfg.sac.reward_function == 'leptin_data':
            self.reward_function = LeptinDataReward2D()
        else:
            self.reward_function = UnSupervisedReward(env=self)

    def execute_action(self, actions, logg_vals=None, post_stats=False, post_images=False):
        self.current_edge_weights = actions.squeeze()

        self.current_soln = self.get_current_soln(self.current_edge_weights)

        if self.cfg.sac.reward_function == 'artificial_cells' or self.cfg.sac.reward_function == 'leptin_data':
            reward = []
            sp_reward = self.reward_function(self.current_soln.long(), self.init_sp_seg.long(), dir_edges=self.edge_ids,
                                             res=500)
            for i, sz in enumerate(self.cfg.sac.s_subgraph):
                reward.append(sp_reward[self.edge_ids][:, self.subgraph_indices[i].view(-1, sz)].sum(0).mean(1))
            reward.append(self.last_final_reward)
            self.counter += 1
            if self.counter >= self.cfg.trainer.max_episode_length:
                self.done = True
                self.last_final_reward = sp_reward.mean()
        else:
            self.sg_current_edge_weights = []
            for i, sz in enumerate(self.cfg.sac.s_subgraph):
                self.sg_current_edge_weights.append(
                    self.current_edge_weights[self.subgraph_indices[i].view(-1, sz)])
            reward = self.reward_function.get(self.sg_current_edge_weights, self.sg_gt_edges) #self.current_soln)
            reward.append(self.last_final_reward)

            self.counter += 1
            if self.counter >= self.cfg.trainer.max_episode_length:
                self.done = True
                self.last_final_reward = self.reward_function.get_global(self.current_edge_weights, self.gt_edge_weights)

        total_reward = 0
        for _rew in reward:
            total_reward += _rew.mean().item()
        total_reward /= len(self.cfg.sac.s_subgraph)

        if self.writer is not None:
            self.writer.add_scalar("step/avg_return", total_reward, self.writer_counter.value())
            if post_images:
                self.writer.add_histogram("step/pred_mean", self.current_edge_weights.view(-1).cpu().numpy(), self.writer_counter.value() // 10)
                fig, axes = plt.subplots(2, 3, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
                axes[0, 0].imshow(self.gt_seg[0].cpu().squeeze())
                axes[0, 0].set_title('gt')
                axes[0, 1].imshow(self.raw[0].cpu().permute(1, 2, 0).squeeze())
                axes[0, 1].set_title('raw image')
                axes[0, 2].imshow(cm.prism(self.init_sp_seg[0].cpu() / self.init_sp_seg[0].max().item()))
                axes[0, 2].set_title('superpixels')
                axes[1, 0].imshow(cm.prism(self.gt_soln[0].cpu()/self.gt_soln[0].max().item()))
                axes[1, 0].set_title('gt')
                axes[1, 1].imshow(pca_project(get_angles(self.embeddings)[0].detach().cpu().numpy()))
                axes[1, 1].set_title('embed')
                axes[1, 2].imshow(cm.prism(self.current_soln[0].cpu()/self.current_soln[0].max().item()))
                axes[1, 2].set_title('prediction')
                self.writer.add_figure("image/state", fig, self.writer_counter.value() // 10)
            if logg_vals is not None and post_stats:
                for key, val in logg_vals.items():
                    self.writer.add_scalar("step/" + key, val, self.writer_counter.value())
            self.writer_counter.increment()

        self.acc_reward.append(total_reward)
        return self.get_state(), reward

    def get_state(self):
        return self.State(self.current_node_embeddings, self.edge_ids, self.edge_angles, self.sup_masses, self.subgraph_indices, self.sep_subgraphs, self.counter, self.gt_edge_weights)

    def update_data(self, raw, gt, edge_ids, gt_edges, sp_seg, **kwargs):
        bs = raw.shape[0]
        dev = raw.device
        self.gt_seg, self.init_sp_seg = gt.squeeze(1), sp_seg.squeeze(1)
        self.raw = raw
        self.embeddings = self.embedding_net(raw).detach()
        subgraphs, self.sep_subgraphs = [], []
        edge_angles, sup_masses, sup_com = zip(*[get_angles_smass_in_rag(edge_ids[i], self.init_sp_seg[i]) for i in range(bs)])
        self.edge_angles, self.sup_masses, self.sup_com = torch.cat(edge_angles).unsqueeze(-1), torch.cat(sup_masses).unsqueeze(-1), torch.cat(sup_com)

        _subgraphs, _sep_subgraphs = find_dense_subgraphs([eids.transpose(0, 1).cpu().numpy() for eids in edge_ids], self.cfg.sac.s_subgraph)
        _subgraphs = [torch.from_numpy(sg.astype(np.int64)).to(dev).permute(2, 0, 1) for sg in _subgraphs]
        _sep_subgraphs = [torch.from_numpy(sg.astype(np.int64)).to(dev).permute(2, 0, 1) for sg in _sep_subgraphs]

        self.n_nodes = [eids.max() + 1 for eids in edge_ids]
        self.edge_ids, (self.n_offs, self.e_offs) = collate_edges(edge_ids)
        self.dir_edge_ids = torch.cat([self.edge_ids, torch.stack([self.edge_ids[1], self.edge_ids[0]], dim=0)], dim=1)
        for i in range(len(self.cfg.sac.s_subgraph)):
            subgraphs.append(torch.cat([sg + self.n_offs[i] for i, sg in enumerate(_subgraphs[i*bs:(i+1)*bs])], -2).flatten(-2, -1))
            self.sep_subgraphs.append(torch.cat(_sep_subgraphs[i*bs:(i+1)*bs], -2).flatten(-2, -1))

        self.subgraphs = subgraphs
        self.subgraph_indices = get_edge_indices(self.edge_ids, subgraphs)

        self.gt_edge_weights = torch.cat(gt_edges)
        self.gt_soln = self.get_current_soln(self.gt_edge_weights)
        self.sg_gt_edges = [self.gt_edge_weights[sg].view(-1, sz) for sz, sg in
                            zip(self.cfg.sac.s_subgraph, self.subgraph_indices)]

        self.current_edge_weights = torch.ones(self.edge_ids.shape[1], device=self.edge_ids.device) / 2

        # get embedding agglomeration over each superpixel
        self.current_node_embeddings = torch.cat([self.embedding_net.get_mean_sp_embedding(embed, sp) for embed, sp
                                                  in zip(self.embeddings, self.init_sp_seg)], dim=0)

        return

    def get_batched_actions_from_global_graph(self, actions):
        b_actions = torch.zeros(size=(self.edge_ids.shape[1],))
        other = torch.zeros_like(self.subgraph_indices)
        for i in range(self.edge_ids.shape[1]):
            mask = (self.subgraph_indices == i)
            num = mask.float().sum()
            b_actions[i] = torch.where(mask, actions.float(), other.float()).sum() / num
        return b_actions

    def get_current_soln(self, edge_weights):
        import h5py
        p_min = 0.001
        p_max = 1.
        segmentations = []
        for i in range(1, len(self.e_offs)):
            probs = edge_weights[self.e_offs[i-1]:self.e_offs[i]]
            edges = self.edge_ids[:, self.e_offs[i-1]:self.e_offs[i]] - self.n_offs[i-1]
            costs = (p_max - p_min) * probs + p_min
            # probabilities to costs
            costs = (torch.log((1. - costs) / costs)).detach().cpu().numpy()
            graph = nifty.graph.undirectedGraph(self.n_nodes[i-1])
            graph.insertEdges(edges.T.cpu().numpy())

            node_labels = elf.segmentation.multicut.multicut_kernighan_lin(graph, costs)

            mc_seg = torch.zeros_like(self.init_sp_seg[i-1])
            unique_lbls = np.unique(node_labels.astype(np.int)).tolist()
            lbl_dict = dict(zip(unique_lbls, np.arange(len(unique_lbls)).astype(np.int).tolist()))
            for j, lbl in enumerate(node_labels.astype(np.int)):
                mc_seg += (self.init_sp_seg[i-1] == j).long() * lbl_dict[lbl]

            segmentations.append(mc_seg)
        return torch.stack(segmentations, dim=0)

    def get_node_gt(self):
        b_node_seg = torch.zeros(self.n_offs[-1], device=self.gt_seg.device)
        for i, (sp_seg, gt) in enumerate(zip(self.init_sp_seg, self.gt_seg)):
            for node_it in range(self.n_nodes[i]):
                nums = torch.bincount(((sp_seg == node_it).long() * (gt.long() + 1)).view(-1))
                b_node_seg[node_it + self.n_offs[i]] = nums[1:].argmax() - 1
        return b_node_seg

    def get_mws(self, affinities):
        affinities[self.sep_chnl:] *= -1
        affinities[self.sep_chnl:] += +1
        affinities[self.sep_chnl:] *= 1.
        affinities[:self.sep_chnl] /= 1.
        affinities = np.clip(affinities, 0, 1)
        #
        node_labeling = compute_mws_segmentation(affinities, self.aff_offsets, self.sep_chnl)
        rag = elf.segmentation.features.compute_rag(np.expand_dims(node_labeling, axis=0))
        edges = rag.uvIds().squeeze().astype(np.int)
        return node_labeling, edges

    def reset(self):
        self.done = False
        self.acc_reward = []
        self.counter = 0


