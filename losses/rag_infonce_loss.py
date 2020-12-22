import torch.nn as nn
import torch
import nifty
import nifty.graph.agglo as nagglo
import numpy as np


class RagInfoNceLoss(nn.Module):

    def __init__(self, tau, similarity):
        super(RagInfoNceLoss, self).__init__()
        self.tau = tau
        self.similarity = similarity

    def forward(self, embeddings, sp_seg, edges):
        # expecting first two args as N, C, D, H, W and third as 2, L. First should be l2-normalized w.r.t. C
        assert embeddings.ndim == sp_seg.ndim
        mask = torch.zeros((int(sp_seg.max()) + 1, ) + sp_seg.shape, dtype=torch.int8, device=sp_seg.device)
        mask.scatter_(0, sp_seg[None].long(), 1)
        masked_embeddings = mask * embeddings[None]
        n_pix_per_sp = mask.flatten(1).sum(1)
        sp_means = masked_embeddings.transpose(1, 2).flatten(2).sum(2) / n_pix_per_sp[:, None]

        intra_sp_sim = self.similarity(sp_means[..., None, None, None], masked_embeddings.transpose(1, 2), dim=1)[mask.bool()] / self.tau
        edge_feats = sp_means[edges]
        inter_sp_sim = self.similarity(edge_feats[0], edge_feats[1], dim=1, kd=False) / self.tau

        intra_sp_sim = intra_sp_sim.exp()
        inter_sp_sim = inter_sp_sim.exp()

        loss = intra_sp_sim / torch.cat((intra_sp_sim, inter_sp_sim), 0).sum()
        loss = - loss.log()
        return loss.mean()
