import torch.nn as nn
import torch
from utils.affinities import get_valid_edges
from skimage.filters import gaussian
import numpy as np


class AffinityContrastive(nn.Module):

    def __init__(self, delta_var, delta_dist, distance, alpha=1.0, beta=1.0):
        super(AffinityContrastive, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.alpha = alpha
        self.beta = beta
        self.sep_chnl = 3
        self.sigma = 1.2
        self.overseg_factor = 1.
        self.offs = [[0, -1], [-1, 0], [-1, -1], [0, -3], [-3, 0], [-3, -3]]
        self.distance = distance

    def get_naive_affinities(self, raw, offsets):
        """get naive pixel affinities based on differences in pixel intensities."""
        affinities = []
        for i, off in enumerate(offsets):
            rolled = torch.roll(raw, tuple(-np.array(off)), (-2, -1))
            dist = torch.norm(raw - rolled, dim=0)
            affinities.append(dist / dist.max())
        return torch.stack(affinities)

    def forward(self, embeddings, raw, **kwargs):
        embeddings = embeddings.squeeze(2)
        cum_loss = []
        for s_embeddings, s_raw in zip(embeddings, raw):
            affs = self.get_naive_affinities(torch.from_numpy(gaussian(s_raw.permute(1, 2, 0).cpu(), self.sigma, multichannel=True)).to(s_raw.device).permute(2, 0, 1), self.offs)

            affs *= -1
            affs += +1
            # scale affinities in order to get an oversegmentation
            affs[:self.sep_chnl] /= self.overseg_factor
            affs[self.sep_chnl:] *= self.overseg_factor
            affs = torch.clamp(affs, 0, 1)

            loss = torch.tensor([0.0], device=s_embeddings.device)
            masks = torch.from_numpy(get_valid_edges([len(self.offs)] + list(s_embeddings.shape[-2:]), self.offs)).to(s_embeddings.device)
            for i, (off, aff, mask) in enumerate(zip(self.offs, affs, masks)):
                rolled = torch.roll(s_embeddings, tuple(-np.array(off)), (-2, -1))
                dist = self.distance(s_embeddings, rolled, dim=0, kd=False)

                aff = aff - aff.min()
                aff = aff / aff.max()

                dist = (dist * (aff - 0.5)) * mask

                loss = loss + dist.mean()

            cum_loss.append(loss)

        return torch.stack(cum_loss).mean()
