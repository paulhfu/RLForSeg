import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.general import pca_project_1d, plt_bar_plot
from models.message_passing import NodeConv, EdgeConv, EdgeConvNoNodes


class Gcnn(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_hidden_layer, hl_factor, device, name, writer=None, start_bn_nl=False):
        super(Gcnn, self).__init__()
        self.name = name
        self.device = device

        self.writer = writer
        self.writer_counter = 0
        self.n_in_channels = n_in_channels
        self.node_conv1 = NodeConv(n_in_channels, n_in_channels,
                                   n_hidden_layer=n_hidden_layer, hl_factor=hl_factor, start_bn_nl=start_bn_nl)
        self.edge_conv1 = EdgeConv(n_in_channels, n_in_channels, use_init_edge_feats=True, n_init_edge_channels=1,
                                    n_hidden_layer=n_hidden_layer, hl_factor=hl_factor)
        self.node_conv2 = NodeConv(n_in_channels, n_in_channels,
                                   n_hidden_layer=n_hidden_layer, hl_factor=hl_factor)
        self.edge_conv2 = EdgeConv(n_in_channels, n_in_channels, use_init_edge_feats=True,
                                   n_init_edge_channels=n_in_channels, n_hidden_layer=n_hidden_layer,
                                   hl_factor=hl_factor)
        self.node_conv3 = NodeConv(n_in_channels, n_in_channels,
                                   n_hidden_layer=n_hidden_layer, hl_factor=hl_factor)
        self.edge_conv3 = EdgeConv(n_in_channels, n_out_channels, use_init_edge_feats=True,
                                   n_init_edge_channels=n_in_channels, n_hidden_layer=n_hidden_layer,
                                   hl_factor=hl_factor)

    def forward(self, node_features, edge_index, angles, gt_edges, post_input=False):

        node_features = self.node_conv1(node_features, edge_index)
        edge_features, side_loss_1 = self.edge_conv1(node_features, edge_index, angles)

        node_features = node_features + self.node_conv2(node_features, edge_index)
        _edge_features, side_loss_2 = self.edge_conv2(node_features, edge_index, edge_features)
        edge_features = _edge_features + edge_features

        node_features = node_features + self.node_conv3(node_features, edge_index)
        edge_features, side_loss_3 = self.edge_conv3(node_features, edge_index, edge_features)

        side_loss = (side_loss_1 + side_loss_2 + side_loss_3) / 3

        if self.writer is not None and post_input and edge_features.shape[1] > 3:
            plt.clf()
            pca_proj_edge_fe = pca_project_1d(edge_features.detach().squeeze().cpu().numpy())
            pca_proj_edge_fe -= pca_proj_edge_fe.min()
            pca_proj_edge_fe /= pca_proj_edge_fe.max()
            selected_edges = torch.multinomial(gt_edges+0.3, 20).cpu().numpy()
            values = np.concatenate([gt_edges[selected_edges].unsqueeze(0).cpu().numpy(), pca_proj_edge_fe[:, selected_edges]], axis=0)
            fig = plt_bar_plot(values, labels=['GT', 'PC1', 'PC2', 'PC3'])
            self.writer.add_figure("bar/embed_edge_features/" + self.name, fig, self.writer_counter)
            self.writer_counter += 1

        return edge_features, side_loss


class QGcnn(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_hidden_layer, hl_factor, device, name, writer=None, start_bn_nl=False):
        super(QGcnn, self).__init__()
        self.name = name
        self.device = device

        self.writer = writer
        self.writer_counter = 0
        self.n_in_channels = n_in_channels
        self.node_conv1 = NodeConv(n_in_channels, n_in_channels,
                                   n_hidden_layer=n_hidden_layer, hl_factor=hl_factor, start_bn_nl=start_bn_nl)
        self.edge_conv1 = EdgeConv(n_in_channels, n_in_channels * 2, use_init_edge_feats=True,
                                    n_init_edge_channels=2, n_hidden_layer=n_hidden_layer, hl_factor=hl_factor)
        self.node_conv2 = NodeConv(n_in_channels, n_in_channels,
                                   n_hidden_layer=n_hidden_layer, hl_factor=hl_factor)
        self.edge_conv2 = EdgeConv(n_in_channels, n_out_channels, use_init_edge_feats=True,
                                    n_init_edge_channels=n_in_channels * 2, n_hidden_layer=n_hidden_layer,
                                   hl_factor=hl_factor)

    def forward(self, node_features, edge_index, angles, gt_edges, actions, post_input=False):

        node_features = self.node_conv1(node_features, edge_index)
        edge_features, side_loss_1 = self.edge_conv1(node_features, edge_index, torch.cat([actions, angles], dim=-1))

        node_features = self.node_conv2(node_features, edge_index)
        edge_features, side_loss_2 = self.edge_conv2(node_features, edge_index, edge_features)

        side_loss = (side_loss_1 + side_loss_2) / 2

        if self.writer is not None and post_input and edge_features.shape[1] > 3:
            pca_proj_edge_fe = pca_project_1d(edge_features.detach().squeeze().cpu().numpy())
            pca_proj_edge_fe -= pca_proj_edge_fe.min()
            pca_proj_edge_fe /= pca_proj_edge_fe.max()
            selected_edges = torch.multinomial(gt_edges+0.3, 20).cpu().numpy()
            values = np.concatenate([gt_edges[selected_edges].unsqueeze(0).cpu().numpy(), pca_proj_edge_fe[:, selected_edges]], axis=0)
            fig = plt_bar_plot(values, labels=['GT', 'PC1', 'PC2', 'PC3'])
            self.writer.add_figure("bar/embed_edge_features/" + self.name, fig, self.writer_counter)
            self.writer_counter += 1

        return edge_features, side_loss


class GlobalEdgeGcnn(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, n_conv_its, hl_factor, device):
        super(GlobalEdgeGcnn, self).__init__()
        self.device = device

        self.n_in_channels = n_in_channels
        self.init_conv = EdgeConvNoNodes()
        self.node_conv = []
        for i in range(n_conv_its):
            self.node_conv.append(NodeConv(n_in_channels, n_in_channels, n_hidden_layer=0, hl_factor=hl_factor))
            super(GlobalEdgeGcnn, self).add_module(f"node_conv_{i}", self.node_conv[-1])

        self.edge_conv = EdgeConv(n_in_channels, n_out_channels, use_init_edge_feats=False, n_hidden_layer=0)

    def forward(self, edge_features, edge_index):

        node_features = self.init_conv(edge_index, edge_features)
        for conv in self.node_conv:
            node_features = node_features + conv(node_features, edge_index)
        edge_features, side_loss = self.edge_conv(node_features, edge_index)

        return edge_features, side_loss

