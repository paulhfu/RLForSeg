import torch

class TemporalSineEncoding(object):

    def __init__(self, max_step, size):
        """
        uniquely encodes a time-step into sine and cosine-frequencies
        :param max_step: maximum discrete time step
        :param size: dim of the encoding vector
        """
        super(TemporalSineEncoding, self).__init__()
        self.max_step = max_step
        self.size = size

    def __call__(self, step, device):
        dims = torch.arange(1, self.size + 1, device=device)
        ret = torch.zeros_like(dims).float()

        ret[0::2] = torch.sin(step/(self.max_step**(dims[0::2]/self.size)))
        ret[1::2] = torch.cos(step/(self.max_step**(dims[1::2]/self.size)))

        return ret


if __name__=="__main__":
    import matplotlib.pyplot as plt
    # do some tests to figure out what frequencies to use.
    n_dim = 8
    max_step = 10

    for i in range(n_dim):
        if i % 2 == 0:
            plt.plot(torch.sin((torch.arange(max_step * 100) / 100) / (max_step**(i/n_dim))))
        else:
            plt.plot(torch.cos((torch.arange(max_step * 100) / 100) / (max_step**(i/n_dim))))

    plt.show()