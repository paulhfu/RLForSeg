import torch
from scipy import interpolate
import numpy as np

class Polygon2d(object):

    def __init__(self, polygonal_chain):
        """
        This class represents polygonal chain, polygons are stored as polygonal chains as well as their projection to a tangent space similar to the one described in here: https://www.researchgate.net/publication/274677198_A_FAST_MATCHING_APPROACH_OF_POLYGON_FEATURES/fulltext/57e99f3208aed0a291304636/A-FAST-MATCHING-APPROACH-OF-POLYGON-FEATURES.pdf
        This implementation stores the tangent angles in a relative manner (w.r.t the predecessor tangent)
        :param polygonal_chain: torch.Tensor of shape (n, 2) where n is the number of vertices of the polygon
        """
        super(Polygon2d, self).__init__()
        dev = polygonal_chain.device
        self.poly_chain = polygonal_chain[:-1]  # polychains unsually end where they start, we want every vertex to be unique in the chain

        face_vecs = self.poly_chain - torch.roll(self.poly_chain, (-1, ), (0, ))
        self.face_lengths_fwd = torch.norm(face_vecs, dim=1)
        self.face_lengths_fwd /= self.face_lengths_fwd.sum()
        self.face_lengths_bwd = torch.flip(self.face_lengths_fwd, dims=(0,))

        normed_face_vecs = face_vecs / torch.norm(face_vecs, dim=1, keepdim=True)

        angles = []
        for nfv in (normed_face_vecs, torch.flip(normed_face_vecs, dims=(0,))):
            dot = (nfv * torch.roll(nfv, (-1, ), (0, ))).sum(1)
            padded, rolled_padded = torch.zeros((nfv.shape[0], 3), device=dev), torch.zeros((nfv.shape[0], 3), device=dev)
            padded[:, 0:2] = nfv
            rolled_padded[:, :2] = torch.roll(nfv, (-1, ), (0, ))
            sign = torch.sign(torch.cross(padded, rolled_padded, dim=1)[:, -1])

            angles.append(torch.acos(dot) * sign)

        self.angles_fwd = angles[0]
        self.angles_bwd = angles[1]
        self.all_tfs = None

    def get_turning_functions(self, n, res):
        """
        Calculates the turning functions of this polygon starting at the nth vertex in both directions
        :param n: starting vertex id
        :param res: resolution of the function
        :return: "res" function values for both directions on the domain [0, 1]
        """
        x_fwd = torch.zeros((self.face_lengths_fwd.shape[0] + 1,))
        x_bwd, y_fwd, y_bwd = x_fwd.clone(), x_fwd.clone(), x_fwd.clone()

        x_fwd[1:] = torch.cumsum(torch.roll(self.face_lengths_fwd, (-n,), (0,)), dim=0)
        x_bwd[1:] = torch.cumsum(torch.flip(torch.roll(self.face_lengths_fwd, (-n,), (0,)), dims=(0,)), dim=0)

        y_fwd[:-1] = torch.cumsum(torch.roll(self.angles_fwd, (-n,), (0,)), dim=0)
        y_bwd[:-1] = torch.cumsum(torch.roll(self.angles_bwd, (-n,), (0,)), dim=0)
        y_fwd[-1] = y_fwd[-2]
        y_bwd[-1] = y_bwd[-2]

        tfs_x = np.linspace(0, 1, res)

        tfs_y_fwd = torch.from_numpy(interpolate.interp1d(x_fwd.cpu(), y_fwd.cpu(), kind='previous')(tfs_x)).to(self.poly_chain.device)
        tfs_y_bwd = torch.from_numpy(interpolate.interp1d(x_bwd.cpu(), y_bwd.cpu(), kind='previous')(tfs_x)).to(self.poly_chain.device)

        return torch.stack([tfs_y_bwd, tfs_y_fwd])


    def distance(self, other, res=100):
        """
        calculates the distance (similarity) of two Polygons by calculating the max value of the area that is enclosed
        by the two turning functions. Of the other polygon only one turning function is considered
        which is compared to all turning functions of this polygon
        :param other: other Polygon
        :param res: number of function values of the turning function (domain is [0, 1])
        :return: similarity score
        """

        if self.all_tfs is None:
            self.all_tfs = torch.cat([self.get_turning_functions(n, res) for n in range(len(self.angles_fwd))], dim=0)

        other_tfs = other.get_turning_functions(0, res)[0][None]
        score = ((other_tfs - self.all_tfs) * (1/res)).abs().sum(1).min()
        return score




