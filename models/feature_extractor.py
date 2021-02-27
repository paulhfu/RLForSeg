import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from utils.general import pca_project
from utils.distances import CosineDistance
from models.unet3d.model import UNet2D

class FeExtractor(nn.Module):
    def __init__(self, backbone_cfg, distance, device=None, writer=None):
        """ Be aware that the underlying embedding space that this feature extractor assumes imploys the cosine distance"""
        super(FeExtractor, self).__init__()
        self.embed_model = UNet2D(**backbone_cfg)

        self.device = device
        self.writer = writer
        self.writer_counter = 0
        self.distance = distance

    def forward(self, raw, post_input=False):
        ret = self.embed_model(raw.unsqueeze(2)).squeeze(2)
        if self.writer is not None and post_input:
            self.post_pca(ret[0])
        if isinstance(self.distance, CosineDistance):
            return ret / torch.clamp(torch.norm(ret, dim=1, keepdim=True), min=1e-10)
        return ret

    # def get_node_features(self, features, sp_indices):
    #     # cannot get the masks with scattering, is too expensive
    #     sp_feat_vecs = torch.empty((len(sp_indices), features.shape[0])).to(self.device).float()
    #     for i, sp in enumerate(sp_indices):
    #         nsp = len(sp)
    #         assert nsp > 0
    #         pix_features = features[:, sp[:, -2], sp[:, -1]].T
    #         pix_features = pix_features[torch.randperm(pix_features.shape[0])]
    #         if nsp > 2:
    #             ind = list(range(0, (nsp//self.max_p) * self.max_p + 1, self.max_p))
    #             ind = [0] if ind == [] else ind
    #             ind = ind + [ind[-1] + nsp % self.max_p] if nsp % self.max_p > 0 else ind
    #             if len(ind) > 2 and ind[-1] - ind[-2] < self.max_p//2:
    #                 ind[-2] -= self.max_p//2
    #
    #             ttl_distances = torch.cat([self.get_weights(pix_features[ind[i]:ind[i+1]]) for i in range(len(ind)-1)])
    #
    #             ttl_distances = ttl_distances / ttl_distances.sum()  # sum up to one
    #             sp_feat_vecs[i] = (ttl_distances[..., None] * pix_features).sum(0)
    #         else:
    #             sp_feat_vecs[i] = pix_features.mean(0)
    #
    #     return sp_feat_vecs
    #
    # def get_weights(self, features):
    #     feature_matrix = features.expand((features.shape[0],) + features.shape)
    #     ttl_distances = self.distance(feature_matrix, feature_matrix.transpose(0, 1), dim=-1, kd=False)
    #     ttl_distances = 1 / ttl_distances.mean(0)
    #     return ttl_distances

    def get_mean_sp_embedding_chunked(self, embeddings, supix, chunks=1):
        """
        This average reduction scheme implements a weighted averaging where the weight distribution is given by the
        normed similarity of each sample to the mean in each superpixel
        """
        dev = embeddings.device
        mask = torch.zeros((int(supix.max()) + 1,) + supix.size(), device=dev).scatter_(0, supix[None], 1)
        slc_sz = mask.shape[0] // chunks
        slices = [slice(slc_sz*step, slc_sz*(step+1), 1) for step in range(chunks)]
        if mask.shape[0] != chunks * slc_sz:
            slices.append(slice(slc_sz*chunks, mask.shape[0], 1))
        sp_embeddings = [self.get_mean_sp_embedding(embeddings, mask[slc]) for slc in slices]

        return torch.cat(sp_embeddings, dim=0)

    def get_mean_sp_embedding(self, embeddings, mask):
        masses = mask.flatten(1).sum(-1)
        masked_embeddings = embeddings[:, None] * mask[None]
        means = masked_embeddings.flatten(2).sum(-1) / masses[None]
        if isinstance(self.distance, CosineDistance):
            means = means / torch.clamp(torch.norm(means, dim=0, keepdim=True), min=1e-10)  # normalize since we use cosine distance
            probs = self.distance.similarity(masked_embeddings, means[..., None, None], dim=0, kd=False)
            probs = probs * mask
            probs = probs / probs.flatten(1).sum(-1)[..., None, None]  # get the probabilities for the embeddings distribution
            sp_embeddings = (masked_embeddings * probs[None]).flatten(2).sum(-1)
            return (sp_embeddings / torch.clamp(torch.norm(sp_embeddings, dim=0, keepdim=True), min=1e-10)).T
        return means.T

    def post_pca(self, features, tag="image/embedding_proj", writer=True):
        plt.clf()
        fig = plt.figure(frameon=False)
        plt.imshow(pca_project(features.detach().cpu().numpy()))
        if writer:
            self.writer.add_figure(tag, fig, self.writer_counter)
            self.writer_counter += 1
        else:
            plt.show()
