import torch
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions import Normal
import math

class ShiftedSigmoidTransform(SigmoidTransform):

    def __init__(self, shift, factor):
        self.shift = shift
        self.factor = factor
        super().__init__()

    def __eq__(self, other):
        return isinstance(other, SigmoidTransform)

    def _call(self, x):
        return super(ShiftedSigmoidTransform, self)._call(x) * self.factor + self.shift

    def _inverse(self, y):
        return super(ShiftedSigmoidTransform, self)._inverse((y - self.shift) / self.factor)

    def log_abs_det_jacobian(self, x, y):
        return super(ShiftedSigmoidTransform, self).log_abs_det_jacobian(x, y - self.shift)

class SigmNorm(TransformedDistribution):

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = Normal(loc, scale)
        transforms = [SigmoidTransform()]
        self._mean = None
        super().__init__(self.base_dist, transforms)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    import sys

    num = 1000
    x = torch.linspace(-50, 50, num)
    y1 = -F.softplus(-x) - F.softplus(x)
    y2 = torch.log(torch.sigmoid(x)*(1-torch.sigmoid(x)))

    plt.plot(x, y1)
    plt.show()
    plt.plot(x, y2)
    plt.show()

    x = torch.linspace(0, 1, num)
    y3 = SigmNorm(torch.tensor([+50.] * num), torch.tensor([-2.] * num).exp()).log_prob(x).exp()
    plt.plot(x, y3.exp())
    plt.show()

    print(sys.float_info)
    loc = torch.ones(100, dtype=torch.float) * 100
    scale = (torch.arange(100, dtype=loc.dtype) + 1) / 30
    dist = SigmNorm(loc, scale)
    samples = dist.rsample()
    lp = dist.log_prob(samples)
    a=1

