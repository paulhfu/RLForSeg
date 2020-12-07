import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from utils.general import pca_project
from models.unet3d.model import UNet2D

class FeExtractor(nn.Module):
    def __init__(self, n_channels=1, n_classes=10, delta=1.5, device=None, writer=None, p_norm=2, max_p_in_distmat=1000):
        super(FeExtractor, self).__init__()
        self.embed_model = UNet2D(n_channels, n_classes, final_sigmoid=False, num_levels=5)

        self.device = device
        self.writer = writer
        self.writer_counter = 0
        self.p_norm = p_norm
        self.delta = delta
        self.max_p = max_p_in_distmat
        self.pw_dist = torch.nn.PairwiseDistance()

    def forward(self, raw, post_input=False):
        ret = self.embed_model(raw.unsqueeze(2)).squeeze(2)
        if self.writer is not None and post_input:
            self.post_pca(ret[0])
        return ret

    def get_node_features(self, features, sp_indices):
        sp_feat_vecs = torch.empty((len(sp_indices), features.shape[0])).to(self.device).float()
        for i, sp in enumerate(sp_indices):
            nsp = len(sp)
            assert nsp > 0
            pix_features = features[:, sp[:, -2], sp[:, -1]].T
            pix_features = pix_features[torch.randperm(pix_features.shape[0])]
            if nsp > 2:
                ind = list(range(0, (nsp//self.max_p) * self.max_p + 1, self.max_p))
                ind = [0] if ind == [] else ind
                ind = ind + [ind[-1] + nsp % self.max_p] if nsp % self.max_p > 0 else ind
                if len(ind) > 2 and ind[-1] - ind[-2] < self.max_p//2:
                    ind[-2] -= self.max_p//2

                ttl_distances = torch.cat([self.get_weights(pix_features[ind[i]:ind[i+1]]) for i in range(len(ind)-1)])

                ttl_distances = ttl_distances / ttl_distances.sum()  # sum up to one
                sp_feat_vecs[i] = (ttl_distances[..., None] * pix_features).sum(0)
            else:
                sp_feat_vecs[i] = pix_features.mean(0)

        return sp_feat_vecs

    def get_weights(self, features):
        feature_matrix = features.expand((features.shape[0],) + features.shape)
        ttl_distances = torch.norm(feature_matrix - feature_matrix.transpose(0, 1), p=self.p_norm, dim=-1)
        ttl_distances = 1 / ttl_distances.mean(0)
        return ttl_distances

    def post_pca(self, features, tag="image/embedding_proj"):
        plt.clf()
        fig = plt.figure(frameon=False)
        plt.imshow(pca_project(features.detach().squeeze().cpu().numpy()))
        self.writer.add_figure(tag, fig, self.writer_counter)
        self.writer_counter += 1
