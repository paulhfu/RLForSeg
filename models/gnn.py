import numpy as np
import torch
import wandb
import torch.nn as nn
import matplotlib.pyplot as plt

from utils.general import pca_project_1d, plt_bar_plot
from models.message_passing import NodeConv, EdgeConv, EdgeConvNoNodes


class EdgeGnn(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_hidden_layer, hl_factor, distance, device, name,
                 depth, normalize_input, start_bn_nl=False):
        super(EdgeGnn, self).__init__()
        self.name = name
        self.device = device

        self.n_in_channels = n_in_channels
        self.depth = depth
        if depth == 1:
            self.node_conv1 = NodeConv(n_in_channels, n_in_channels, distance=distance, normalize_input=False,
                                       n_hidden_layer=n_hidden_layer, hl_factor=hl_factor, start_bn_nl=start_bn_nl)
            self.edge_conv1 = EdgeConv(n_in_channels, n_out_channels, use_init_edge_feats=True, n_init_edge_channels=1,
                                       n_hidden_layer=n_hidden_layer, hl_factor=hl_factor)
        if depth == 2:
            self.node_conv1 = NodeConv(n_in_channels, n_in_channels, distance=distance, normalize_input=False,
                                       n_hidden_layer=n_hidden_layer, hl_factor=hl_factor, start_bn_nl=start_bn_nl)
            self.edge_conv1 = EdgeConv(n_in_channels, n_in_channels, use_init_edge_feats=True, n_init_edge_channels=1,
                                       n_hidden_layer=n_hidden_layer, hl_factor=hl_factor)
            self.node_conv2 = NodeConv(n_in_channels, n_in_channels, distance=distance, normalize_input=normalize_input,
                                       n_hidden_layer=n_hidden_layer, hl_factor=hl_factor)
            self.edge_conv2 = EdgeConv(n_in_channels, n_out_channels, use_init_edge_feats=True,
                                       n_init_edge_channels=n_in_channels, n_hidden_layer=n_hidden_layer,
                                       hl_factor=hl_factor)
        if depth == 3:
            self.node_conv1 = NodeConv(n_in_channels, n_in_channels, distance=distance, normalize_input=False,
                                       n_hidden_layer=n_hidden_layer, hl_factor=hl_factor, start_bn_nl=start_bn_nl)
            self.edge_conv1 = EdgeConv(n_in_channels, n_in_channels, use_init_edge_feats=True, n_init_edge_channels=1,
                                       n_hidden_layer=n_hidden_layer, hl_factor=hl_factor)
            self.node_conv2 = NodeConv(n_in_channels, n_in_channels, distance=distance, normalize_input=normalize_input,
                                       n_hidden_layer=n_hidden_layer, hl_factor=hl_factor)
            self.edge_conv2 = EdgeConv(n_in_channels, n_in_channels, use_init_edge_feats=True,
                                       n_init_edge_channels=n_in_channels, n_hidden_layer=n_hidden_layer,
                                       hl_factor=hl_factor)
            self.node_conv3 = NodeConv(n_in_channels, n_in_channels, distance=distance, normalize_input=normalize_input,
                                       n_hidden_layer=n_hidden_layer, hl_factor=hl_factor)
            self.edge_conv3 = EdgeConv(n_in_channels, n_out_channels, use_init_edge_feats=True,
                                       n_init_edge_channels=n_in_channels, n_hidden_layer=n_hidden_layer,
                                       hl_factor=hl_factor)

    def forward(self, node_features, edge_index, angles, gt_edges, post_data=False):
        side_loss = torch.tensor([0.0], device=node_features.device)
        node_features = self.node_conv1(node_features, edge_index)
        edge_features, sl = self.edge_conv1(node_features, edge_index, angles)
        side_loss = side_loss + sl
        if self.depth > 1:
            node_features = node_features + self.node_conv2(node_features, edge_index)
            _edge_features, sl = self.edge_conv2(node_features, edge_index, edge_features)
            side_loss = side_loss + sl
            if self.depth == 2:
                edge_features = _edge_features
        if self.depth > 2:
            edge_features = _edge_features + edge_features
            node_features = node_features + self.node_conv3(node_features, edge_index)
            edge_features, sl = self.edge_conv3(node_features, edge_index, edge_features)
            side_loss = side_loss + sl

        side_loss = side_loss / self.depth

        if post_data and edge_features.shape[1] > 3 and gt_edges is not None:
            plt.clf()
            pca_proj_edge_fe = pca_project_1d(edge_features.detach().squeeze().cpu().numpy())
            pca_proj_edge_fe -= pca_proj_edge_fe.min()
            pca_proj_edge_fe /= pca_proj_edge_fe.max()
            selected_edges = torch.multinomial(gt_edges + 0.3, 20).cpu().numpy()
            values = np.concatenate(
                [gt_edges[selected_edges].unsqueeze(0).cpu().numpy(), pca_proj_edge_fe[:, selected_edges]], axis=0)
            fig = plt_bar_plot(values, labels=['GT', 'PC1', 'PC2', 'PC3'])
            wandb.log({"bar": [wandb.Image(fig, caption="embed_edge_features" + self.name)]})

        return edge_features, side_loss


class NodeGnn(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_hidden_layer, hl_factor, distance, device, name,
                 start_bn_nl=False):
        super(NodeGnn, self).__init__()
        self.name = name
        self.device = device

        self.n_in_channels = n_in_channels
        self.node_conv1 = NodeConv(n_in_channels, n_in_channels, distance=distance, normalize_input=False,
                                   n_hidden_layer=n_hidden_layer, hl_factor=hl_factor, start_bn_nl=start_bn_nl)
        self.node_conv2 = NodeConv(n_in_channels, n_in_channels, distance=distance, normalize_input=True,
                                   n_hidden_layer=n_hidden_layer, hl_factor=hl_factor)
        self.node_conv3 = NodeConv(n_in_channels, n_in_channels, distance=distance, normalize_input=True,
                                   n_hidden_layer=n_hidden_layer, hl_factor=hl_factor)
        self.node_conv4 = NodeConv(n_in_channels, n_out_channels, distance=distance, normalize_input=True,
                                   n_hidden_layer=n_hidden_layer, hl_factor=hl_factor)

    def forward(self, node_features, edge_index, angles, gt_edges, post_input=False):
        node_features = self.node_conv1(node_features, edge_index)
        node_features = node_features + self.node_conv2(node_features, edge_index)
        node_features = node_features + self.node_conv3(node_features, edge_index)
        return self.node_conv4(node_features, edge_index), torch.tensor([0.0], device=node_features.device)


class QGnn(nn.Module):
    def __init__(self, n_node_in_features, n_edge_in_features, n_out_features, n_hidden_layer, hl_factor, distance,
                 device, name, depth, normalize_input, start_bn_nl=False):
        super(QGnn, self).__init__()
        self.name = name
        self.device = device
        self.depth = depth
        if depth==1:
            self.node_conv1 = NodeConv(n_node_in_features, n_node_in_features, distance=distance, normalize_input=False,
                                       n_hidden_layer=n_hidden_layer, hl_factor=hl_factor, start_bn_nl=start_bn_nl)
            self.edge_conv1 = EdgeConv(n_node_in_features, n_out_features, use_init_edge_feats=True,
                                       n_init_edge_channels=n_edge_in_features, n_hidden_layer=n_hidden_layer,
                                       hl_factor=hl_factor)
        if depth==2:
            self.node_conv1 = NodeConv(n_node_in_features, n_node_in_features, distance=distance, normalize_input=False,
                                       n_hidden_layer=n_hidden_layer, hl_factor=hl_factor, start_bn_nl=start_bn_nl)
            self.edge_conv1 = EdgeConv(n_node_in_features, n_node_in_features * 2, use_init_edge_feats=True,
                                       n_init_edge_channels=n_edge_in_features, n_hidden_layer=n_hidden_layer,
                                       hl_factor=hl_factor)
            self.node_conv2 = NodeConv(n_node_in_features, n_node_in_features, distance=distance, normalize_input=normalize_input,
                                       n_hidden_layer=n_hidden_layer, hl_factor=hl_factor)
            self.edge_conv2 = EdgeConv(n_node_in_features, n_out_features, use_init_edge_feats=True,
                                       n_init_edge_channels=n_node_in_features * 2, n_hidden_layer=n_hidden_layer,
                                       hl_factor=hl_factor)

    def forward(self, node_features, edge_features, edge_index, gt_edges, post_data=False):
        node_features = self.node_conv1(node_features, edge_index)
        edge_features, side_loss_1 = self.edge_conv1(node_features, edge_index, edge_features)
        if self.depth==2:
            node_features = self.node_conv2(node_features, edge_index)
            edge_features, side_loss_2 = self.edge_conv2(node_features, edge_index, edge_features)

        side_loss = side_loss_1

        if post_data and edge_features.shape[1] > 3 and gt_edges is not None:
            pca_proj_edge_fe = pca_project_1d(edge_features.detach().squeeze().cpu().numpy())
            pca_proj_edge_fe -= pca_proj_edge_fe.min()
            pca_proj_edge_fe /= pca_proj_edge_fe.max()
            selected_edges = torch.multinomial(gt_edges + 0.3, 20).cpu().numpy()
            values = np.concatenate(
                [gt_edges[selected_edges].unsqueeze(0).cpu().numpy(), pca_proj_edge_fe[:, selected_edges]], axis=0)
            fig = plt_bar_plot(values, labels=['GT', 'PC1', 'PC2', 'PC3'])
            wandb.log({"bar": [wandb.Image(fig, caption="embed_edge_features" + self.name)]})

        return edge_features, side_loss


class GlobalEdgeGnn(nn.Module):
    def __init__(self, n_in_features, n_out_features, n_conv_its, hl_factor, device):
        super(GlobalEdgeGnn, self).__init__()
        self.device = device

        self.n_in_channels = n_in_features
        self.init_conv = EdgeConvNoNodes()
        self.node_conv = []
        for i in range(n_conv_its):
            self.node_conv.append(NodeConv(n_in_features, n_in_features, n_hidden_layer=0, hl_factor=hl_factor))
            super(GlobalEdgeGnn, self).add_module(f"node_conv_{i}", self.node_conv[-1])

        self.edge_conv = EdgeConv(n_in_features, n_out_features, use_init_edge_feats=False, n_hidden_layer=0)

    def forward(self, edge_features, edge_index):

        node_features = self.init_conv(edge_index, edge_features)
        for conv in self.node_conv:
            node_features = node_features + conv(node_features, edge_index)
        edge_features, side_loss = self.edge_conv(node_features, edge_index)

        return edge_features, side_loss
