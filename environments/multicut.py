from matplotlib import cm
import numpy as np
import torch
from utils import general
import collections
import matplotlib.pyplot as plt
from utils.reward_functions import UnSupervisedReward, SubGraphDiceReward
from utils.graphs import collate_edges, get_edge_indices, get_angles_smass_in_rag
from rag_utils import find_dense_subgraphs
import elf
from elf.segmentation.multicut import multicut_kernighan_lin
import nifty
from affogato.segmentation import compute_mws_segmentation

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
        self.max_p = torch.nn.MaxPool2d(3, padding=1, stride=1)
        self.aff_offsets = [[0, -1], [-1, 0], [-1, -1], [-8, 0], [0, -8]]
        self.sep_chnl = 3

        if self.cfg.sac.reward_function == 'sub_graph_dice':
            self.reward_function = SubGraphDiceReward()
        else:
            self.reward_function = UnSupervisedReward(env=self)

    def execute_action(self, actions, logg_vals=None, post_stats=False):
        self.current_edge_weights = actions

        self.sg_current_edge_weights = []
        for i, sz in enumerate(self.cfg.sac.s_subgraph):
            self.sg_current_edge_weights.append(
                self.current_edge_weights[self.subgraph_indices[i].view(-1, sz)])

        self.current_soln = self.get_current_soln(self.current_edge_weights)
        reward = self.reward_function.get(self.sg_current_edge_weights, self.sg_gt_edges) #self.current_soln)

        self.counter += 1
        if self.counter >= self.cfg.trainer.max_episode_length:
            self.done = True

        total_reward = 0
        for _rew in reward:
            total_reward += _rew.mean().item()
        total_reward /= len(self.cfg.sac.s_subgraph)

        if self.writer is not None and post_stats:
            self.writer.add_scalar("step/avg_return", total_reward, self.writer_counter.value())
            if self.writer_counter.value() % 10 == 0:
                self.writer.add_histogram("step/pred_mean", self.current_edge_weights.view(-1).cpu().numpy(), self.writer_counter.value() // 10)
                fig, (a1, a2, a3, a4) = plt.subplots(1, 4, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
                a1.imshow(self.raw[0].cpu().permute(1,2,0).squeeze())
                a1.set_title('raw image')
                a2.imshow(cm.prism(self.init_sp_seg[0].cpu() / self.init_sp_seg[0].max().item()))
                a2.set_title('superpixels')
                a3.imshow(cm.prism(self.gt_soln[0].cpu()/self.gt_soln[0].max().item()))
                a3.set_title('gt')
                a4.imshow(cm.prism(self.current_soln[0].cpu()/self.current_soln[0].max().item()))
                a4.set_title('prediction')
                self.writer.add_figure("image/state", fig, self.writer_counter.value() // 10)
                self.embedding_net.post_pca(self.embeddings[0].cpu(), tag="image/pix_embedding_proj")
            self.writer.add_scalar("step/gt_mean", self.gt_edge_weights.mean().item(), self.writer_counter.value())
            self.writer.add_scalar("step/gt_std", self.gt_edge_weights.std().item(), self.writer_counter.value())
            if logg_vals is not None:
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
        self.init_sp_seg_edge = torch.cat([(-self.max_p(-self.init_sp_seg) != self.init_sp_seg).float(), (self.max_p(self.init_sp_seg) != self.init_sp_seg).float()], 1)

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

        stacked_superpixels = [torch.zeros((int(sp.max()+1), ) + sp.shape, device=self.device).scatter_(0, sp[None].long(), 1) for sp in self.init_sp_seg]
        self.sp_indices = [[torch.nonzero(sp, as_tuple=False) for sp in stacked_superpixel] for stacked_superpixel in stacked_superpixels]

        # get embedding agglomeration over each superpixel
        node_feats = []
        for i, sp_ind in enumerate(self.sp_indices):
            n_f = self.embedding_net.get_node_features(self.embeddings[i], sp_ind)
            node_feats.append(n_f)
        self.current_node_embeddings = torch.cat(node_feats, dim=0)

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
            for j, lbl in enumerate(node_labels):
                mc_seg += (self.init_sp_seg[i-1] == j).float() * lbl

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


