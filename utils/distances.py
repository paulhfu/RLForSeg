import torch

class CosineDistance(object):

    def __init__(self):
        super(CosineDistance, self).__init__()
        self.has_normed_similarity = True

    def __call__(self, x, y, dim, kd=True, **kwargs):
        """x and y need to be l2 normed along dim
            result is in [0, 2]"""
        return 1.0 - (x * y).sum(dim=dim, keepdim=kd)

    def similarity(self, x, y, dim, kd=True, **kwargs):
        """x and y need to be l2 normed along dim
            result is in [0, 2]"""
        return 1.0 + (x * y).sum(dim=dim, keepdim=kd)


class LpDistance(object):

    def __init__(self):
        super(LpDistance, self).__init__()
        self.has_normed_similarity = False

    def __call__(self, x, y, dim, kd=True, ord=None, **kwargs):
        return torch.linalg.norm(x - y, dim=dim, keepdim=kd, ord=ord)

