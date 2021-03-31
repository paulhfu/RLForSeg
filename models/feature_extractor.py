import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from multiprocessing import Lock
from utils.distances import CosineDistance
from models.unet3d.model import UNet2D

class FeExtractor(nn.Module):
    def __init__(self, backbone_cfg, distance, device=None):
        """ Be aware that the underlying embedding space that this feature extractor assumes imploys the cosine distance"""
        super(FeExtractor, self).__init__()
        self.embed_model = UNet2D(**backbone_cfg)

        self.device = device
        self.distance = distance
        self.fwd_mtx = Lock()

    def forward(self, raw):
        self.fwd_mtx.acquire()
        try:
            ret = self.embed_model(raw.unsqueeze(2)).squeeze(2)
        finally:
            self.fwd_mtx.release()
        if isinstance(self.distance, CosineDistance):
            ret = ret / (torch.norm(ret, dim=1, keepdim=True) + 1e-10)
        return ret

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

