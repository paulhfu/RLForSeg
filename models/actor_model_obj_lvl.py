import torch
import numpy as np
import wandb
import torch.nn as nn
from models.gnn import EdgeGnn, NodeGnn, QNodeGnn, GlobalNodeGnn
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
        embed_dim = self.cfg.dim_embeddings + 3

        self.actor = PolicyNet(embed_dim, 2, cfg.gnn_n_hidden, cfg.gnn_hl_factor, distance, device, False,
                               cfg.gnn_act_depth, cfg.gnn_act_norm_inp)

    def forward(self, state, actions, expl_action, post_data, policy_opt, return_node_features):
        state = self.StateClass(*state)
        # node_features = state.node_embeddings
        node_features = torch.cat((state.node_embeddings, state.sp_feat), 1)
        edge_index = torch.cat([state.edge_ids, torch.stack([state.edge_ids[1], state.edge_ids[0]], dim=0)],
                               dim=1)  # gcnn expects two directed edges for one undirected edge
        if actions is None:
            with torch.set_grad_enabled(policy_opt):
                out, side_loss = self.actor(node_features, edge_index, state.edge_angles, state.gt_edge_weights,
                                            post_data)
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

            # this means exploration
            if return_node_features:
                return dist, actions, node_features
            return dist, actions, side_loss


class PolicyNet(torch.nn.Module):
    def __init__(self, n_in_features, n_classes, n_hidden_layer, hl_factor, distance, device, node_actions, depth,
                 normalize_input):
        super(PolicyNet, self).__init__()
        if node_actions:
            self.gcn = NodeGnn(n_in_features, n_classes, n_hidden_layer, hl_factor, distance, device, "actor")
        else:
            self.gcn = EdgeGnn(n_in_features, n_classes, n_hidden_layer, hl_factor, distance, device, "actor", depth,
                               normalize_input)

    def forward(self, node_features, edge_index, angles, gt_edges, post_data):
        actor_stats, side_loss = self.gcn(node_features, edge_index, angles, gt_edges, post_data)
        return actor_stats, side_loss


