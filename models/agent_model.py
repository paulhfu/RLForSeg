import torch
import numpy as np
import wandb
import torch.nn as nn
from models.gnn import EdgeGnn, NodeGnn, QGnn, GlobalEdgeGnn
from utils.distances import CosineDistance
from utils.sigmoid_normal import SigmNorm

class Agent(torch.nn.Module):
    def __init__(self, cfg, StateClass, distance, device):
        super(Agent, self).__init__()
        self.cfg = cfg
        self.std_bounds = self.cfg.std_bounds
        self.mu_bounds = self.cfg.mu_bounds
        self.device = device
        self.StateClass = StateClass
        self.distance = distance

        self.actor = PolicyNet(self.cfg.dim_embeddings, 2, cfg.gnn_n_hidden,
                               cfg.gnn_hl_factor, distance, device, False, cfg.gnn_act_depth, cfg.gnn_act_norm_inp)
        self.critic = DoubleQValueNet(self.cfg.s_subgraph, self.cfg.dim_embeddings,
                                      1, 1, cfg.gnn_n_hidden, cfg.gnn_hl_factor,
                                      distance, device, False, cfg.gnn_crit_depth, cfg.gnn_crit_norm_inp)
        self.critic_tgt = DoubleQValueNet(self.cfg.s_subgraph, self.cfg.dim_embeddings,
                                          1, 1, cfg.gnn_n_hidden, cfg.gnn_hl_factor,
                                          distance, device, False, cfg.gnn_crit_depth, cfg.gnn_crit_norm_inp)

        self.log_alpha = torch.tensor([np.log(self.cfg.init_temperature)] * len(self.cfg.s_subgraph)).to(device)
        self.log_alpha.requires_grad = True

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @alpha.setter
    def alpha(self, value):
        self.log_alpha = torch.tensor(np.log(value)).to(self.device)
        self.log_alpha.requires_grad = True

    def forward(self, state, actions, post_data, policy_opt, return_node_features):
        state = self.StateClass(*state)
        node_features = state.node_embeddings
        # node_features = torch.cat((state.node_embeddings, state.sup_masses), 1)
        edge_index = torch.cat([state.edge_ids, torch.stack([state.edge_ids[1], state.edge_ids[0]], dim=0)], dim=1)  # gcnn expects two directed edges for one undirected edge
        if actions is None:
            with torch.set_grad_enabled(policy_opt):
                out, side_loss = self.actor(node_features, edge_index, state.edge_angles, state.gt_edge_weights, post_data)
                mu, std = out.chunk(2, dim=-1)
                mu, std = mu.contiguous(), std.contiguous()

                if post_data:
                    wandb.log({"logits/loc": wandb.Histogram(mu.view(-1).detach().cpu().numpy())})
                    wandb.log({"logits/scale": wandb.Histogram(std.view(-1).detach().cpu().numpy())})

                std = self.std_bounds[0] + 0.5 * (self.std_bounds[1] - self.std_bounds[0]) * (torch.tanh(std) + 1)
                mu = self.mu_bounds[0] + 0.5 * (self.mu_bounds[1] - self.mu_bounds[0]) * (torch.tanh(mu) + 1)

                dist = SigmNorm(mu, std)
                actions = dist.rsample()

            q1, q2, sl = self.critic_tgt(node_features, actions, edge_index, state.edge_angles, state.subgraph_indices,
                                         state.sep_subgraphs, state.gt_edge_weights, post_data)
            side_loss = (side_loss + sl) / 2
            if policy_opt:
                return dist, q1, q2, actions, side_loss
            else:
                # this means either exploration or critic opt
                if return_node_features:
                    return dist, q1, q2, actions, None, side_loss, node_features
                return dist, q1, q2, actions, None, side_loss

        q1, q2, side_loss = self.critic(node_features, actions, edge_index, state.edge_angles, state.subgraph_indices,
                                        state.sep_subgraphs, state.gt_edge_weights, post_data)
        return q1, q2, side_loss
    

class PolicyNet(torch.nn.Module):
    def __init__(self, n_in_features, n_classes, n_hidden_layer, hl_factor, distance, device, node_actions, depth, normalize_input):
        super(PolicyNet, self).__init__()
        if node_actions:
            self.gcn = NodeGnn(n_in_features, n_classes, n_hidden_layer, hl_factor, distance, device, "actor")
        else:
            self.gcn = EdgeGnn(n_in_features, n_classes, n_hidden_layer, hl_factor, distance, device, "actor", depth, normalize_input)

    def forward(self, node_features, edge_index, angles, gt_edges, post_data):
        actor_stats, side_loss = self.gcn(node_features, edge_index, angles, gt_edges, post_data)
        return actor_stats, side_loss


class DoubleQValueNet(torch.nn.Module):
    def __init__(self, s_subgraph, n_in_features, n_actions, n_classes, n_hidden_layer, hl_factor, distance, device,
                 node_actions, depth, normalize_input):
        super(DoubleQValueNet, self).__init__()

        self.s_subgraph = s_subgraph
        self.node_actions = node_actions
        n_node_in_features = n_in_features
        n_edge_in_features = n_actions + 1
        if node_actions:
            n_node_in_features += n_actions
            n_edge_in_features = 1

        self.gcn1_1 = QGnn(n_node_in_features, n_edge_in_features, n_node_in_features, n_hidden_layer, hl_factor,
                           distance, device, "critic", depth, normalize_input)
        self.gcn2_1 = QGnn(n_node_in_features, n_edge_in_features, n_node_in_features, n_hidden_layer, hl_factor,
                           distance, device, "critic", depth, normalize_input)

        self.gcn1_2, self.gcn2_2 = [], []

        for i, ssg in enumerate(self.s_subgraph):
            self.gcn1_2.append(GlobalEdgeGnn(n_node_in_features, n_node_in_features, ssg, hl_factor, device))
            self.gcn2_2.append(GlobalEdgeGnn(n_node_in_features, n_node_in_features, ssg, hl_factor, device))
            super(DoubleQValueNet, self).add_module(f"gcn1_2_{i}", self.gcn1_2[-1])
            super(DoubleQValueNet, self).add_module(f"gcn2_2_{i}", self.gcn2_2[-1])

        self.value1 = nn.Sequential(
            torch.nn.BatchNorm1d(n_node_in_features, track_running_stats=False),
            torch.nn.LeakyReLU(),
            nn.Linear(n_node_in_features, hl_factor * 4),
            torch.nn.BatchNorm1d(hl_factor * 4, track_running_stats=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hl_factor * 4, hl_factor * 4),
            torch.nn.BatchNorm1d(hl_factor * 4, track_running_stats=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hl_factor * 4, n_classes),
        )

        self.value2 = nn.Sequential(
            torch.nn.BatchNorm1d(n_node_in_features, track_running_stats=False),
            torch.nn.LeakyReLU(inplace=True),
            nn.Linear(n_node_in_features, hl_factor * 4),
            torch.nn.BatchNorm1d(hl_factor * 4, track_running_stats=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hl_factor * 4, hl_factor * 4),
            torch.nn.BatchNorm1d(hl_factor * 4, track_running_stats=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hl_factor * 4, n_classes),
        )

    def forward(self, node_features, actions, edge_index, angles, sub_graphs, sep_subgraphs, gt_edges, post_data):
        if actions.ndim < 2:
            actions = actions.unsqueeze(-1)
        if self.node_actions:
            node_features = torch.cat([node_features, actions], dim=-1)
            edge_features = angles
        else:
            edge_features = torch.cat([actions, angles], dim=-1)
        _sg_edge_features1, side_loss = self.gcn1_1(node_features, edge_features, edge_index, gt_edges, post_data)

        _sg_edge_features2, _side_loss = self.gcn2_1(node_features, edge_features, edge_index, gt_edges, post_data)
        side_loss += _side_loss

        sg_edge_features1, sg_edge_features2 = [], []
        for i, sg_size in enumerate(self.s_subgraph):
            sub_sg_edge_features1 = _sg_edge_features1[sub_graphs[i]]
            sub_sg_edge_features2 = _sg_edge_features2[sub_graphs[i]]
            sg_edges = torch.cat([sep_subgraphs[i], torch.stack([sep_subgraphs[i][1], sep_subgraphs[i][0]], dim=0)], dim=1)  # gcnn expects two directed edges for one undirected edge

            sub_sg_edge_features1, _side_loss = self.gcn1_2[i](sub_sg_edge_features1, sg_edges)
            side_loss += _side_loss
            sub_sg_edge_features2, _side_loss = self.gcn2_2[i](sub_sg_edge_features2, sg_edges)
            side_loss += _side_loss

            sg_edge_features1.append(self.value1(sub_sg_edge_features1.view(-1, sg_size,
                                                                            sub_sg_edge_features1.shape[-1]).mean(1)).squeeze())
            sg_edge_features2.append(self.value2(sub_sg_edge_features2.view(-1, sg_size,
                                                                            sub_sg_edge_features2.shape[-1]).mean(1)).squeeze())

        return sg_edge_features1, sg_edge_features2, side_loss / 4
