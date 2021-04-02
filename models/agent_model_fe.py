import torch
import numpy as np
import wandb
import torch.nn as nn
from models.gnn import EdgeGnn, NodeGnn, QGnn, GlobalEdgeGnn
from utils.distances import CosineDistance
from models.feature_extractor import FeExtractor
from utils.sigmoid_normal import SigmNorm
from utils.general import get_contour_from_2d_binary
from utils.yaml_conv_parser import dict_to_attrdict


class Agent(torch.nn.Module):
    def __init__(self, cfg, StateClass, distance, device, with_temp=True):
        super(Agent, self).__init__()
        self.cfg = cfg
        self.std_bounds = self.cfg.std_bounds
        self.mu_bounds = self.cfg.mu_bounds
        self.device = device
        self.StateClass = StateClass
        self.distance = distance
        dim_embed = self.cfg.dim_embeddings + 3


        self.actor = PolicyNet(dim_embed, 2, cfg.gnn_n_hl, cfg.gnn_size_hl, distance, cfg, device, False, cfg.gnn_act_depth, cfg.gnn_act_norm_inp)
        self.critic = QValueNet(self.cfg.s_subgraph, dim_embed, 1, 1, cfg.gnn_n_hl, cfg.gnn_size_hl, distance, cfg, device, False, cfg.gnn_crit_depth, cfg.gnn_crit_norm_inp)
        self.critic_tgt = QValueNet(self.cfg.s_subgraph, dim_embed, 1, 1, cfg.gnn_n_hl, cfg.gnn_size_hl, distance, cfg, device, False, cfg.gnn_crit_depth, cfg.gnn_crit_norm_inp)

        self.log_alpha = torch.tensor([np.log(self.cfg.init_temperature)] * len(self.cfg.s_subgraph)).to(device)
        if with_temp:
            self.log_alpha.requires_grad = True

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @alpha.setter
    def alpha(self, value):
        self.log_alpha = torch.tensor(np.log(value)).to(self.device)
        self.log_alpha.requires_grad = True

    def forward(self, state, actions, expl_action, post_data, policy_opt, return_node_features, get_embeddings):
        state = self.StateClass(*state)
        # node_features = state.node_embeddings

        edge_index = torch.cat([state.edge_ids, torch.stack([state.edge_ids[1], state.edge_ids[0]], dim=0)], dim=1)  # gcnn expects two directed edges for one undirected edge
        if actions is None:
            with torch.set_grad_enabled(policy_opt):
                out, side_loss, embeddings, node_features = self.actor(state, edge_index, state.edge_feats, state.gt_edge_weights, post_data)
                mu, std = out.chunk(2, dim=-1)
                mu, std = mu.contiguous(), std.contiguous()

                if post_data:
                    wandb.log({"logits/loc": wandb.Histogram(mu.view(-1).detach().cpu().numpy())})
                    wandb.log({"logits/scale": wandb.Histogram(std.view(-1).detach().cpu().numpy())})

                std = self.std_bounds[0] + 0.5 * (self.std_bounds[1] - self.std_bounds[0]) * (torch.tanh(std) + 1)
                mu = self.mu_bounds[0] + 0.5 * (self.mu_bounds[1] - self.mu_bounds[0]) * (torch.tanh(mu) + 1)

                dist = SigmNorm(mu, std)
                if expl_action is None:
                    actions = dist.rsample()
                else:
                    z = ((expl_action - mu) / std).detach()
                    actions = mu + z * std

            q, sl = self.critic_tgt(state, actions, edge_index, state.edge_feats, state.subgraph_indices,
                                         state.sep_subgraphs, state.gt_edge_weights, post_data)
            side_loss = (side_loss + sl) / 2
            if policy_opt:
                return dist, q, actions, side_loss
            else:
                # this means either exploration or critic opt
                if return_node_features:
                    if get_embeddings:
                        return dist, q, actions, None, side_loss, node_features, embeddings.detach()
                    return dist, q, actions, None, side_loss, node_features
                if get_embeddings:
                    return dist, q, actions, None, side_loss, embeddings.detach()
                return dist, q, actions, None, side_loss

        q, side_loss = self.critic(state, actions, edge_index, state.edge_feats, state.subgraph_indices,
                                        state.sep_subgraphs, state.gt_edge_weights, post_data)
        return q, side_loss


class PolicyNet(torch.nn.Module):
    def __init__(self, n_in_features, n_classes, n_hidden_layer, hl_factor, distance, cfg, device, node_actions, depth, normalize_input):
        super(PolicyNet, self).__init__()

        self.fe_ext = FeExtractor(dict_to_attrdict(cfg.backbone), distance, device)
        # self.fe_ext.embed_model.load_state_dict(torch.load(cfg.fe_model_name))
        self.fe_ext.cuda(device)

        if node_actions:
            self.gcn = NodeGnn(n_in_features, n_classes, n_hidden_layer, hl_factor, distance, device, "actor")
        else:
            self.gcn = EdgeGnn(n_in_features, n_classes, 3, n_hidden_layer, hl_factor, distance, device, "actor", depth, normalize_input)

    def forward(self, state, edge_index, edge_feats, gt_edges, post_data):

        embeddings = self.fe_ext(state.raw)
        # get embedding agglomeration over each superpixel
        node_features = torch.cat([self.fe_ext.get_mean_sp_embedding_chunked(embed, sp, chunks=100)
                                                  for embed, sp in zip(embeddings, state.sp_seg)], dim=0)

        node_features = torch.cat((node_features, state.sp_feat), 1)

        actor_stats, side_loss = self.gcn(node_features, edge_index, edge_feats, gt_edges, post_data)
        return actor_stats, side_loss, embeddings, node_features


class QValueNet(torch.nn.Module):
    def __init__(self, s_subgraph, n_in_features, n_actions, n_classes, n_hidden_layer, hl_factor, distance, cfg, device,
                 node_actions, depth, normalize_input):
        super(QValueNet, self).__init__()

        self.fe_ext = FeExtractor(dict_to_attrdict(cfg.backbone), distance, device)
        # self.fe_ext.embed_model.load_state_dict(torch.load(cfg.fe_model_name))
        self.fe_ext.cuda(device)

        self.s_subgraph = s_subgraph
        self.node_actions = node_actions
        n_node_in_features = n_in_features
        n_edge_in_features = n_actions + 3
        if node_actions:
            n_node_in_features += n_actions
            n_edge_in_features = 1

        self.gcn = QGnn(n_node_in_features, n_edge_in_features, n_node_in_features, n_hidden_layer, hl_factor,
                           distance, device, "critic", depth, normalize_input)

        self.value = []

        for i, ssg in enumerate(self.s_subgraph):
            self.value.append(nn.Sequential(
                torch.nn.BatchNorm1d(n_node_in_features * ssg, track_running_stats=False),
                torch.nn.LeakyReLU(),
                nn.Linear(n_node_in_features * ssg, hl_factor),
                torch.nn.BatchNorm1d(hl_factor, track_running_stats=False),
                nn.LeakyReLU(inplace=True),
                nn.Linear(hl_factor, hl_factor),
                torch.nn.BatchNorm1d(hl_factor, track_running_stats=False),
                nn.LeakyReLU(inplace=True),
                nn.Linear(hl_factor, n_classes),
            ))
            super(QValueNet, self).add_module(f"value{i}", self.value[-1])

    def forward(self, state, actions, edge_index, edge_feat, sub_graphs, sep_subgraphs, gt_edges, post_data):

        embeddings = self.fe_ext(state.raw)

        # get embedding agglomeration over each superpixel
        node_features = torch.cat([self.fe_ext.get_mean_sp_embedding_chunked(embed, sp, chunks=100)
                                                  for embed, sp in zip(embeddings, state.sp_seg)], dim=0)

        node_features = torch.cat((node_features, state.sp_feat), 1)

        if actions.ndim < 2:
            actions = actions.unsqueeze(-1)
        if self.node_actions:
            node_features = torch.cat([node_features, actions], dim=-1)
            edge_features = edge_feat
        else:
            edge_features = torch.cat([actions, edge_feat], dim=-1)

        edge_feats, side_loss = self.gcn(node_features, edge_features, edge_index, gt_edges, post_data)

        sg_edge_features = []
        for i, ssg in enumerate(self.s_subgraph):
            sg_edge_feats = edge_feats[sub_graphs[i]].view(-1, ssg, edge_feats.shape[-1])
            sg_edge_feats = sg_edge_feats.view(sg_edge_feats.shape[0], ssg * sg_edge_feats.shape[-1])
            sg_edge_features.append(self.value[i](sg_edge_feats).squeeze())

        return sg_edge_features, side_loss
