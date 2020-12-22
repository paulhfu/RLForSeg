import torch.nn as nn
import torch
import nifty
import nifty.graph.agglo as nagglo
import numpy as np


class RagContrastiveLoss(nn.Module):

    def __init__(self, delta_var, delta_dist, distance, alpha=1.0, beta=1.0):
        super(RagContrastiveLoss, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.alpha = alpha
        self.beta = beta
        self.distance = distance

    def forward(self, embeddings, sp_seg, edges, **kwargs):
        # expecting first two args as N, C, D, H, W and third as 2, L. First should be l2-normalized w.r.t. C
        assert embeddings.ndim == sp_seg.ndim
        C = int(sp_seg.max()) + 1
        mask = torch.zeros((C, ) + sp_seg.shape, dtype=torch.int8, device=sp_seg.device)
        mask.scatter_(0, sp_seg[None].long(), 1)
        masked_embeddings = mask * embeddings[None]
        n_pix_per_sp = mask.flatten(1).sum(1, keepdim=True)
        sp_means = masked_embeddings.transpose(1, 2).flatten(2).sum(2) / n_pix_per_sp

        intra_sp_dist = self.distance(sp_means[:, None, :, None, None, None], masked_embeddings, dim=2)
        intra_sp_dist = torch.clamp(intra_sp_dist - self.delta_var, min=0) / n_pix_per_sp[..., None, None, None, None]
        intra_sp_dist = intra_sp_dist[mask.bool()].sum() / C

        edge_feats = sp_means[edges]
        inter_sp_dist = self.distance(edge_feats[0], edge_feats[1], dim=1, kd=False)
        inter_sp_dist = torch.clamp(self.delta_dist - inter_sp_dist, min=0)
        inter_sp_dist = inter_sp_dist.sum() / edges.shape[1]

        loss = self.alpha * inter_sp_dist + self.beta * intra_sp_dist
        return loss.mean()
