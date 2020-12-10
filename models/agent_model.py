import torch
import torch.nn as nn
from models.gnn import EdgeGnn, NodeGnn, QGnn, GlobalEdgeGnn
from utils.sigmoid_normal import SigmNorm
import numpy as np

class AgentEdge(torch.nn.Module):
    def __init__(self, cfg, device, writer=None):
        super(AgentEdge, self).__init__()
        self.writer = writer
        self.cfg = cfg
        self.std_bounds = self.cfg.sac.diag_gaussian_actor.std_bounds
        self.mu_bounds = self.cfg.sac.diag_gaussian_actor.mu_bounds
        self.sample_offset = self.cfg.sac.diag_gaussian_actor.sample_offset
        self.device = device
        self.writer_counter = 0

        self.actor = PolicyNet(self.cfg.fe.n_embedding_features+2, self.cfg.sac.n_actions, cfg.model.n_hidden,
                               cfg.model.hl_factor, device, writer, "nodes" in self.cfg.gen.env)
        self.critic = DoubleQValueNet(self.cfg.sac.s_subgraph, self.cfg.fe.n_embedding_features + 2, 1,
                                      cfg.model.n_hidden, cfg.model.hl_factor, device, writer)
        self.critic_tgt = DoubleQValueNet(self.cfg.sac.s_subgraph, self.cfg.fe.n_embedding_features + 2, 1,
                                          cfg.model.n_hidden, cfg.model.hl_factor, device, writer)

        self.log_alpha = torch.tensor([np.log(self.cfg.sac.init_temperature)] * len(self.cfg.sac.s_subgraph)).to(device)
        self.log_alpha.requires_grad = True

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @alpha.setter
    def alpha(self, value):
        self.log_alpha = torch.tensor(np.log(value)).to(self.device)
        self.log_alpha.requires_grad = True

    def forward(self, state, actions, post_input, policy_opt, embeddings_opt, return_node_features):
        node_features, edge_index, angles, sup_masses, sub_graphs, sep_subgraphs, round_n, gt_edges = state
        round_n /= self.cfg.trainer.max_episode_length
        node_features = torch.cat((node_features, sup_masses, torch.ones_like(sup_masses) * round_n), 1)
        edge_index = torch.cat([edge_index, torch.stack([edge_index[1], edge_index[0]], dim=0)], dim=1)  # gcnn expects two directed edges for one undirected edge
        if actions is None:
            with torch.set_grad_enabled(policy_opt):
                out, side_loss = self.actor(node_features, edge_index, angles, gt_edges, post_input)
                mu, std = out.chunk(2, dim=-1)
                mu, std = mu.squeeze(), std.squeeze()

                if post_input and self.writer is not None:
                    self.writer.add_histogram("hist_logits/loc", mu.view(-1).detach().cpu().numpy(), self.writer_counter)
                    self.writer.add_histogram("hist_logits/scale", std.view(-1).detach().cpu().numpy(), self.writer_counter)
                    self.writer_counter += 1

                std = self.std_bounds[0] + 0.5 * (self.std_bounds[1] - self.std_bounds[0]) * (torch.tanh(std) + 1)
                mu = self.mu_bounds[0] + 0.5 * (self.mu_bounds[1] - self.mu_bounds[0]) * (torch.tanh(mu) + 1)

                dist = SigmNorm(mu, std, sample_offset=self.sample_offset)
                actions = dist.rsample()

            q1, q2, sl = self.critic_tgt(node_features, actions, edge_index, angles, sub_graphs, sep_subgraphs, gt_edges, post_input)
            side_loss = (side_loss + sl) / 2
            if policy_opt:
                return dist, q1, q2, actions, side_loss
            else:
                # this means either exploration,critic opt or embedding opt
                if return_node_features:
                    return dist, q1, q2, actions, None, side_loss, node_features
                return dist, q1, q2, actions, None, side_loss

        q1, q2, side_loss = self.critic(node_features, actions, edge_index, angles, sub_graphs, sep_subgraphs, gt_edges, post_input)
        return q1, q2, side_loss
    

class PolicyNet(torch.nn.Module):
    def __init__(self, n_in_features, n_classes, n_hidden_layer, hl_factor, device, writer, node_actions):
        super(PolicyNet, self).__init__()
        if node_actions:
            self.gcn = EdgeGnn(n_in_features, n_classes, n_hidden_layer, hl_factor, device, "actor", writer)
        else:
            self.gcn = NodeGnn(n_in_features, n_classes, n_hidden_layer, hl_factor, device, "actor", writer)

    def forward(self, node_features, edge_index, angles, gt_edges, post_input):
        actor_stats, side_loss = self.gcn(node_features, edge_index, angles, gt_edges, post_input)
        return actor_stats, side_loss


class DoubleQValueNet(torch.nn.Module):
    def __init__(self, s_subgraph, n_in_features, n_classes, n_hidden_layer, hl_factor, device, writer):
        super(DoubleQValueNet, self).__init__()

        self.s_subgraph = s_subgraph

        self.gcn1_1 = QGnn(n_in_features, n_in_features, n_hidden_layer, hl_factor, device, "critic", writer)
        self.gcn2_1 = QGnn(n_in_features, n_in_features, n_hidden_layer, hl_factor, device, "critic", writer)

        self.gcn1_2, self.gcn2_2 = [], []

        for i, ssg in enumerate(self.s_subgraph):
            self.gcn1_2.append(GlobalEdgeGnn(n_in_features, n_in_features, ssg//2, hl_factor, device))
            self.gcn2_2.append(GlobalEdgeGnn(n_in_features, n_in_features, ssg//2, hl_factor, device))
            super(DoubleQValueNet, self).add_module(f"gcn1_2_{i}", self.gcn1_2[-1])
            super(DoubleQValueNet, self).add_module(f"gcn2_2_{i}", self.gcn2_2[-1])

        self.value1 = nn.Sequential(
            torch.nn.BatchNorm1d(n_in_features, track_running_stats=False),
            torch.nn.LeakyReLU(),
            nn.Linear(n_in_features, hl_factor * 4),
            torch.nn.BatchNorm1d(hl_factor * 4, track_running_stats=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hl_factor * 4, hl_factor * 4),
            torch.nn.BatchNorm1d(hl_factor * 4, track_running_stats=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hl_factor * 4, n_classes),
        )

        self.value2 = nn.Sequential(
            torch.nn.BatchNorm1d(n_in_features, track_running_stats=False),
            torch.nn.LeakyReLU(inplace=True),
            nn.Linear(n_in_features, hl_factor * 4),
            torch.nn.BatchNorm1d(hl_factor * 4, track_running_stats=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hl_factor * 4, hl_factor * 4),
            torch.nn.BatchNorm1d(hl_factor * 4, track_running_stats=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hl_factor * 4, n_classes),
        )

    def forward(self, node_features, actions, edge_index, angles, sub_graphs, sep_subgraphs, gt_edges, post_input):
        actions = actions.unsqueeze(-1)
        _sg_edge_features1, side_loss = self.gcn1_1(node_features, edge_index, angles, gt_edges, actions, post_input)

        _sg_edge_features2, _side_loss = self.gcn2_1(node_features, edge_index, angles, gt_edges, actions, post_input)
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
