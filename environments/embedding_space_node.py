from matplotlib import cm
import numpy as np
import torch
import elf
import nifty
import collections
import matplotlib.pyplot as plt
from utils.reward_functions import UnSupervisedReward, SubGraphDiceReward
from utils.graphs import collate_edges, get_edge_indices, get_angles_smass_in_rag
from utils.general import pca_project_1d
from rag_utils import find_dense_subgraphs
import nifty.graph.agglo as nagglo
from scipy.cluster.hierarchy import linkage, fcluster

class EmbeddingSpaceEnvNodeBased():

    State = collections.namedtuple("State", ["node_embeddings", "edge_ids", "edge_angles", "sup_masses", "subgraph_indices", "sep_subgraphs", "subgraphs", "round_n", "gt_edge_weights"])

    def __init__(self, embedding_net, cfg, device, writer=None, writer_counter=None):
        super(EmbeddingSpaceEnvNodeBased, self).__init__()
        self.embedding_net = embedding_net
        self.reset()
        self.cfg = cfg
        self.device = device
        self.writer = writer
        self.writer_counter = writer_counter
        self.max_p = torch.nn.MaxPool2d(3, padding=1, stride=1)

        if self.cfg.sac.reward_function == 'sub_graph_dice':
            self.reward_function = SubGraphDiceReward()
        else:
            self.reward_function = UnSupervisedReward(env=self)

        if self.cfg.gen.embedding_dist == "cosine":
            self.cluster_policy = nagglo.cosineDistNodeAndEdgeWeightedClusterPolicy
        else:
            self.cluster_policy = nagglo.nodeAndEdgeWeightedClusterPolicy

    def execute_action(self, actions, logg_vals=None, post_stats=False):

        self.current_node_embeddings += actions
        if self.cfg.gen.embedding_dist == 'cosine':
            self.current_node_embeddings /= (torch.norm(self.current_node_embeddings, dim=-1) + 1e-6)

        self.current_soln, node_labeling = self.get_soln_graph_clustering(self.current_node_embeddings)

        sg_edge_weights = []
        for i, sz in enumerate(self.cfg.sac.s_subgraph):
            sg_ne = node_labeling[self.subgraphs[i].view(2, -1, sz)]
            sg_edge_weights.append((sg_ne[0] == sg_ne[1]).float())

        reward = self.reward_function.get(sg_edge_weights, self.sg_gt_edges) #self.current_soln)

        self.counter += 1
        if self.counter >= self.cfg.trainer.max_episode_length:
            self.done = True

        total_reward = 0
        for _rew in reward:
            total_reward += _rew.mean().item()
        total_reward /= len(self.cfg.sac.s_subgraph)

        if self.writer is not None and post_stats:
            self.writer.add_scalar("step/avg_return", total_reward, self.writer_counter.value())
            if self.writer_counter.value() % 20 == 0:
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
                self.writer.add_figure("image/shift_proj", self.vis_node_actions(actions.cpu(), 0), self.writer_counter.value() // 10)
                self.embedding_net.post_pca(self.embeddings[0].cpu(), tag="image/pix_embedding_proj")
                self.embedding_net.post_pca(self.current_node_embeddings[:self.n_offs[1]][self.init_sp_seg[0].long(), :].T.cpu(),
                                            tag="image/node_embedding_proj")

            if logg_vals is not None:
                for key, val in logg_vals.items():
                    self.writer.add_scalar("step/" + key, val, self.writer_counter.value())
            self.writer_counter.increment()

        self.acc_reward.append(total_reward)
        return self.get_state(), reward

    def get_state(self):
        return self.State(self.current_node_embeddings, self.edge_ids, self.edge_angles, self.sup_masses, self.subgraph_indices, self.sep_subgraphs, self.subgraphs, self.counter, self.gt_edge_weights)

    def update_data(self, edge_ids, gt_edges, sp_seg, raw, gt, **kwargs):
        bs = len(edge_ids)
        dev = edge_ids[0].device
        subgraphs, self.sep_subgraphs = [], []
        self.gt_seg = gt.squeeze(1)
        self.raw = raw
        self.init_sp_seg = sp_seg.squeeze(1)
        edge_angles, sup_masses, sup_com = zip(*[get_angles_smass_in_rag(edges, sp) for edges, sp in zip(edge_ids, self.init_sp_seg)])
        self.edge_angles, self.sup_masses, self.sup_com = torch.cat(edge_angles).unsqueeze(-1), torch.cat(sup_masses).unsqueeze(-1), torch.cat(sup_com)
        self.init_sp_seg_edge = torch.cat([(-self.max_p(-sp_seg) != sp_seg).float(), (self.max_p(sp_seg) != sp_seg).float()], 1)

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
        self.gt_soln = self.get_mc_soln(self.gt_edge_weights)
        self.sg_gt_edges = [self.gt_edge_weights[sg].view(-1, sz) for sz, sg in
                            zip(self.cfg.sac.s_subgraph, self.subgraph_indices)]

        stacked_superpixels = [torch.zeros((int(sp.max()+1), ) + sp.shape, device=self.device).scatter_(0, sp[None].long(), 1) for sp in self.init_sp_seg]
        self.sp_indices = [[torch.nonzero(sp, as_tuple=False) for sp in stacked_superpixel] for stacked_superpixel in stacked_superpixels]

        self.embeddings = self.embedding_net(self.raw).detach()
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

    def get_soln_free_clustering(self, node_features):
        labels = []
        node_labels = []
        for i, sp_seg in enumerate(self.init_sp_seg):
            single_node_features = node_features[self.n_offs[i]:self.n_offs[i+1]]
            z_linkage = linkage(single_node_features.cpu(), 'ward')
            node_labels.append(fcluster(z_linkage, self.cfg.gen.n_max_object, criterion='maxclust'))
            rag = elf.segmentation.features.compute_rag(np.expand_dims(sp_seg.cpu(), axis=0))
            labels.append(elf.segmentation.features.project_node_labels_to_pixels(rag, node_labels[-1]).squeeze())

        return torch.from_numpy(np.stack(labels).astype(np.float)).to(node_features.device), \
               torch.from_numpy(np.concatenate(node_labels).astype(np.float)).to(node_features.device)

    def get_soln_graph_clustering(self, node_features):
        labels = []
        node_labels = []
        for i, sp_seg in enumerate(self.init_sp_seg):
            single_node_features = node_features[self.n_offs[i]:self.n_offs[i+1]].detach().cpu().numpy()
            rag = nifty.graph.undirectedGraph(single_node_features.shape[0])
            rag.insertEdges((self.edge_ids[:, self.e_offs[i]:self.e_offs[i+1]] - self.n_offs[i]).T.detach().cpu().numpy())

            edge_weights = np.ones(rag.numberOfEdges, dtype=np.int)
            edge_sizes = np.ones(rag.numberOfEdges, dtype=np.int)
            node_sizes = np.ones(rag.numberOfNodes, dtype=np.int)

            policy = nagglo.nodeAndEdgeWeightedClusterPolicy(
                graph=rag,
                edgeIndicators=edge_weights,
                edgeSizes=edge_sizes,
                nodeFeatures=single_node_features,
                nodeSizes=node_sizes,
                numberOfNodesStop=self.cfg.gen.n_max_object,
                beta = 1,
                sizeRegularizer = 0
            )
            clustering = nagglo.agglomerativeClustering(policy)
            clustering.run()

            node_labels.append(clustering.result())
            rag = elf.segmentation.features.compute_rag(np.expand_dims(sp_seg.cpu(), axis=0))
            labels.append(elf.segmentation.features.project_node_labels_to_pixels(rag, node_labels[-1]).squeeze())
        return torch.from_numpy(np.stack(labels).astype(np.float)).to(node_features.device), \
               torch.from_numpy(np.concatenate(node_labels).astype(np.float)).to(node_features.device)

    def get_mc_soln(self, edge_weights):
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

    def vis_node_actions(self, shifts, sb=0):
        plt.clf()
        fig = plt.figure()
        shifts = shifts[self.n_offs[sb]:self.n_offs[sb+1]]
        n = shifts.shape[0]
        proj = pca_project_1d(shifts, 8)
        proj = np.concatenate((proj[:2], proj[2:4], proj[4:6], proj[6:8]), 1)
        colors = n*["b"] + n*["g"] + n*["r"] + n*["c"]
        com = np.round(self.sup_com[self.n_offs[sb]:self.n_offs[sb+1]].cpu())
        com = np.concatenate((com, )*4, 0)
        plt.imshow((self.gt_seg[sb]*(self.init_sp_seg_edge[sb, 0] == 0) + self.init_sp_seg_edge[sb, 0] * 10).cpu())
        plt.quiver(com[:, 1], com[:, 0], proj[0], proj[1], color=colors, width=0.005)
        return fig

    def reset(self):
        self.done = False
        self.acc_reward = []
        self.counter = 0
