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
        self.critic = DoubleQValueNet(embed_dim, 1, 1, cfg.gnn_n_hidden, cfg.gnn_hl_factor, distance, device, False,
                                      cfg.gnn_crit_depth, cfg.gnn_crit_norm_inp)
        self.critic_tgt = DoubleQValueNet(embed_dim,
                                          1, 1, cfg.gnn_n_hidden, cfg.gnn_hl_factor,
                                          distance, device, False, cfg.gnn_crit_depth, cfg.gnn_crit_norm_inp)

        self.log_alpha = torch.tensor(np.log(self.cfg.init_temperature)).to(device)
        self.log_alpha.requires_grad = True

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @alpha.setter
    def alpha(self, value):
        self.log_alpha = torch.tensor(np.log(value)).to(self.device)
        self.log_alpha.requires_grad = True

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

            if policy_opt:
                q1, q2 = self.critic_tgt(node_features, actions, state.edge_ids, state.edge_angles,
                                         state.obj_edge_ind_critic, state.obj_node_mask_critic, state.gt_edge_weights,
                                         post_data)
                return dist, q1, q2, actions, side_loss
            else:
                # this means exploration
                if return_node_features:
                    return dist, actions, node_features
                return dist, actions

        q1, q2 = self.critic(node_features, actions, state.edge_ids, state.edge_angles, state.obj_edge_ind_critic,
                             state.obj_node_mask_critic, state.gt_edge_weights, post_data)
        return q1, q2


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


class DoubleQValueNet(torch.nn.Module):
    def __init__(self, n_in_features, n_actions, n_classes, n_hidden_layer, hl_factor, distance, device,
                 node_actions, depth, normalize_input):
        super(DoubleQValueNet, self).__init__()

        self.node_actions = node_actions
        n_node_in_features = n_in_features
        n_edge_in_features = n_actions + 1
        if node_actions:
            n_node_in_features += n_actions
            n_edge_in_features = 1

        self.gcn1_1 = QNodeGnn(n_node_in_features, n_edge_in_features, n_node_in_features, n_hidden_layer, hl_factor,
                               distance, device, "critic", depth, normalize_input)
        self.gcn2_1 = QNodeGnn(n_node_in_features, n_edge_in_features, n_node_in_features, n_hidden_layer, hl_factor,
                               distance, device, "critic", depth, normalize_input)

        self.gcn1_2 = (GlobalNodeGnn(n_node_in_features, n_node_in_features, 10, hl_factor, device))
        self.gcn2_2 = (GlobalNodeGnn(n_node_in_features, n_node_in_features, 10, hl_factor, device))

        value1 = [nn.LeakyReLU(), nn.Linear(n_node_in_features, 256)]
        for i in range(10):
            value1 += [nn.LeakyReLU(inplace=True), nn.Linear(256, 256)]
        value1 += [nn.LeakyReLU(inplace=True), nn.Linear(256, n_classes)]
        self.value1 = nn.Sequential(*value1)

        value2 = [nn.LeakyReLU(), nn.Linear(n_node_in_features, 256)]
        for i in range(10):
            value2 += [nn.LeakyReLU(inplace=True), nn.Linear(256, 256)]
        value2 += [nn.LeakyReLU(inplace=True), nn.Linear(256, n_classes)]
        self.value2 = nn.Sequential(*value2)

    def forward(self, node_features, actions, edge_ids, angles, obj_edge_ind, obj_node_mask, gt_edges, post_data):
        if actions.ndim < 2:
            actions = actions.unsqueeze(-1)
        if self.node_actions:
            node_features = torch.cat([node_features, actions], dim=-1)
            edge_features = angles
        else:
            edge_features = torch.cat([actions, angles], dim=-1)

        dir_edge_features = torch.cat([edge_features, edge_features], dim=0)
        dir_edge_ids = torch.cat([edge_ids, torch.stack([edge_ids[1], edge_ids[0]], dim=0)], dim=1)

        node_features1 = self.gcn1_1(node_features, dir_edge_features, dir_edge_ids, gt_edges, post_data)
        node_features2 = self.gcn2_1(node_features, dir_edge_features, dir_edge_ids, gt_edges, post_data)

        edge_ids_obj = edge_ids[:, obj_edge_ind]
        dir_edge_ids_obj = torch.cat([edge_ids_obj, torch.stack([edge_ids_obj[1], edge_ids_obj[0]], dim=0)], dim=1)

        node_features1 = self.gcn1_2(node_features1, dir_edge_ids_obj)
        node_features2 = self.gcn2_2(node_features2, dir_edge_ids_obj)

        obj_mass = obj_node_mask.sum(1)
        object_features1 = (node_features1[None] * obj_node_mask[..., None]).sum(1) / obj_mass[:, None]
        object_features2 = (node_features2[None] * obj_node_mask[..., None]).sum(1) / obj_mass[:, None]

        # if object_features1.shape[0] == 1:
        #     for layer in self.value1:
        #         if isinstance(layer, nn.BatchNorm1d):
        #             layer.eval()
        #     for layer in self.value2:
        #         if isinstance(layer, nn.BatchNorm1d):
        #             layer.eval()
        # else:
        #     for layer in self.value1:
        #         if isinstance(layer, nn.BatchNorm1d):
        #             layer.train()
        #     for layer in self.value2:
        #         if isinstance(layer, nn.BatchNorm1d):
        #             layer.train()

        sg_edge_features1 = self.value1(object_features1)
        sg_edge_features2 = self.value2(object_features2)

        return sg_edge_features1, sg_edge_features2
