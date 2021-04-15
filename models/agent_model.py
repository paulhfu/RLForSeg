import torch
import numpy as np
import wandb
import torch.nn as nn
from models.gnn import EdgeGnn, NodeGnn, QGnn, GlobalEdgeGnn
from utils.distances import CosineDistance
from utils.yaml_conv_parser import dict_to_attrdict
from models.feature_extractor import FeExtractor
from utils.sigmoid_normal import SigmNorm
from utils.affinities import get_edge_features_1d, get_affinities_from_embeddings_2d


class Agent(torch.nn.Module):
    def __init__(self, cfg, StateClass, distance, device, with_temp=True):
        super(Agent, self).__init__()
        self.cfg = cfg
        self.std_bounds = self.cfg.std_bounds
        self.mu_bounds = self.cfg.mu_bounds
        self.device = device
        self.StateClass = StateClass
        self.distance = distance
        self.offs = [[1, 0], [0, 1], [2, 0], [0, 2], [4, 0], [0, 4], [16, 0], [0, 16]]

        dim_embed = self.cfg.dim_embeddings + (3 * int(cfg.use_handcrafted_features))

        # self.fe_ext = FeExtractor(dict_to_attrdict(self.cfg.backbone), self.distance, cfg.fe_delta_dist, self.device)
        # self.fe_ext.embed_model.load_state_dict(torch.load(self.cfg.fe_model_name))
        # self.fe_ext.cuda(self.device)
        # for param in self.fe_ext.parameters():
        #     param.requires_grad = False

        self.fe_ext_tgt = FeExtractor(dict_to_attrdict(self.cfg.backbone), self.distance, cfg.fe_delta_dist, self.device)
        self.fe_ext_tgt.embed_model.load_state_dict(torch.load(self.cfg.fe_model_name))
        self.fe_ext_tgt.cuda(self.device)
        for param in self.fe_ext_tgt.parameters():
            param.requires_grad = False

        self.actor = PolicyNet(dim_embed, 2, cfg.gnn_n_hl, cfg.gnn_size_hl, distance, cfg, cfg.gnn_dropout, device, False, cfg.gnn_act_depth, cfg.gnn_act_norm_inp, cfg.n_init_edge_feat)
        self.critic = QValueNet(self.cfg.s_subgraph, dim_embed, 1, 1, cfg.gnn_n_hl, cfg.gnn_size_hl, distance,
                                cfg.gnn_dropout, device, False, cfg.gnn_crit_depth, cfg.gnn_crit_norm_inp,
                                cfg.n_init_edge_feat)
        self.critic_tgt = QValueNet(self.cfg.s_subgraph, dim_embed, 1, 1, cfg.gnn_n_hl, cfg.gnn_size_hl, distance,
                                    cfg.gnn_dropout, device, False, cfg.gnn_crit_depth, cfg.gnn_crit_norm_inp,
                                    cfg.n_init_edge_feat)

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

    def get_features(self, model, state, grad):
        with torch.set_grad_enabled(grad):
            embeddings = model(state.raw)
        embed_affs = get_affinities_from_embeddings_2d(embeddings.detach(), self.offs, model.delta_dist, model.distance)
        embed_dists = [1 - get_edge_features_1d(sp.cpu().numpy(), self.offs, embed_aff.cpu().numpy())[0][:, 0] for sp, embed_aff in zip(state.sp_seg, embed_affs)]
        embed_dists = [torch.from_numpy(embed_dist).to(model.device) for embed_dist in embed_dists]
        embed_dists = torch.cat(embed_dists, 0)[:, None]
        node_features = torch.cat([model.get_mean_sp_embedding_chunked(embed, sp, chunks=100)
                                   for embed, sp in zip(embeddings, state.sp_seg)], dim=0)

        node_features = torch.cat((node_features, state.sp_feat), 1) if self.cfg.use_handcrafted_features else node_features
        edge_features = torch.cat((embed_dists, state.edge_feats), 1).float()

        return node_features, edge_features, embeddings

    def forward(self, state, actions, expl_action, post_data, policy_opt, return_node_features, get_embeddings):
        state = self.StateClass(*state)

        # node_features, edge_features, _ = self.get_features(self.fe_ext, state, grad=False)
        node_features_tgt, edge_features_tgt, embeddings_tgt = self.get_features(self.fe_ext_tgt, state, grad=False)
        node_features, edge_features = node_features_tgt, edge_features_tgt

        edge_index = torch.cat([state.edge_ids, torch.stack([state.edge_ids[1], state.edge_ids[0]], dim=0)], dim=1)  # gcnn expects two directed edges for one undirected edge
        if actions is None:
            with torch.set_grad_enabled(policy_opt):
                out, side_loss = self.actor(node_features, edge_index, edge_features, state.gt_edge_weights, post_data)
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

            q, sl = self.critic_tgt(node_features_tgt, actions, edge_index, edge_features_tgt, state.subgraph_indices,
                                    state.sep_subgraphs, state.gt_edge_weights, post_data)
            side_loss = (side_loss + sl) / 2
            if policy_opt:
                return dist, q, actions, side_loss
            else:
                # this means either exploration or critic opt
                if return_node_features:
                    if get_embeddings:
                        return dist, q, actions, None, side_loss, node_features, embeddings_tgt.detach()
                    return dist, q, actions, None, side_loss, node_features
                if get_embeddings:
                    return dist, q, actions, None, side_loss, embeddings_tgt.detach()
                return dist, q, actions, None, side_loss

        q, side_loss = self.critic(node_features, actions, edge_index, edge_features, state.subgraph_indices,
                                   state.sep_subgraphs, state.gt_edge_weights, post_data)
        return q, side_loss



class PolicyNet(torch.nn.Module):
    def __init__(self, n_in_features, n_classes, n_hidden_layer, hl_factor, distance, cfg, dropout, device, node_actions,
                 depth, normalize_input, n_edge_feat):
        super(PolicyNet, self).__init__()

        self.edge_feat = n_edge_feat is not None
        if node_actions:
            self.gcn = NodeGnn(n_in_features, n_classes, n_hidden_layer, hl_factor, distance, dropout, device, "actor")
        else:
            self.gcn = EdgeGnn(n_in_features, n_classes, self.edge_feat, n_edge_feat, n_hidden_layer,
                               hl_factor, distance, dropout, device, "actor", depth, normalize_input, False)


    def forward(self, node_features, edge_index, edge_features, gt_edges, post_data):
        edge_feats = edge_features if self.edge_feat else None

        actor_stats, side_loss = self.gcn(node_features, edge_index, edge_feats, gt_edges, post_data)
        return actor_stats, side_loss

class QValueNet(torch.nn.Module):
    def __init__(self, s_subgraph, n_in_features, n_actions, n_classes, n_hidden_layer, hl_factor, distance, dropout,
                 device, node_actions, depth, normalize_input, n_edge_feat):
        super(QValueNet, self).__init__()

        self.s_subgraph = s_subgraph
        self.node_actions = node_actions
        self.edge_feat = n_edge_feat is not None
        n_node_in_features = n_in_features
        n_edge_in_features = n_actions + n_edge_feat if self.edge_feat else n_actions
        if node_actions:
            n_node_in_features += n_actions
            n_edge_in_features = n_edge_feat if self.edge_feat else None

        self.gcn = QGnn(n_node_in_features, n_edge_in_features, n_node_in_features, n_hidden_layer, hl_factor,
                        distance, device, "critic", depth, normalize_input, dropout, False)

        self.value = []

        for i, ssg in enumerate(self.s_subgraph):
            self.value.append(nn.Sequential(
                torch.nn.BatchNorm1d(n_node_in_features * ssg, track_running_stats=False),
                torch.nn.LeakyReLU(),
                nn.Linear(n_node_in_features * ssg, hl_factor),
                # nn.Dropout(dropout),
                torch.nn.BatchNorm1d(hl_factor, track_running_stats=False),
                torch.nn.LeakyReLU(),
                nn.Linear(hl_factor, hl_factor),
                # nn.Dropout(dropout),
                torch.nn.BatchNorm1d(hl_factor, track_running_stats=False),
                torch.nn.LeakyReLU(),
                nn.Linear(hl_factor, n_classes),
                # nn.Dropout(dropout),
            ))
            super(QValueNet, self).add_module(f"value{i}", self.value[-1])

    def forward(self, node_features, actions, edge_index, edge_feat, sub_graphs, sep_subgraphs, gt_edges, post_data):
        if actions.ndim < 2:
            actions = actions.unsqueeze(-1)
        if self.node_actions:
            node_features = torch.cat([node_features, actions], dim=-1)
            edge_features = edge_feat
        else:
            edge_features = torch.cat([actions, edge_feat], dim=-1) if self.edge_feat else actions

        edge_feats, side_loss = self.gcn(node_features, edge_features, edge_index, gt_edges, post_data)

        sg_edge_features = []
        for i, ssg in enumerate(self.s_subgraph):
            sg_edge_feats = edge_feats[sub_graphs[i]].view(-1, ssg, edge_feats.shape[-1])
            sg_edge_feats = sg_edge_feats.view(sg_edge_feats.shape[0], ssg * sg_edge_feats.shape[-1])
            sg_edge_features.append(self.value[i](sg_edge_feats).squeeze())

        return sg_edge_features, side_loss
