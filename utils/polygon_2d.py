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

        # get the lengths of each face and normalize them by the polygon perimeter
        face_vecs = self.poly_chain - torch.roll(self.poly_chain, (-1, ), (0, ))
        self.face_lengths_fwd = torch.norm(face_vecs, dim=1) + 1e-6
        self.face_lengths_fwd /= self.face_lengths_fwd.sum()
        self.face_lengths_bwd = torch.flip(self.face_lengths_fwd, dims=(0,))

        normed_face_vecs = face_vecs / (torch.norm(face_vecs, dim=1, keepdim=True) + 1e-6)

        # get the turning  angles at each vertex. turning angles are positive or negative [0, pi] depending in which direction they point
        angles = []
        for nfv in (normed_face_vecs, torch.flip(normed_face_vecs, dims=(0,))):  # get the angles for both walking directions
            # clalculate the angle by the acos of the dot product. determine the turning direction by the sign of the z
            # component of the 0 padded cross product
            dot = (nfv * torch.roll(nfv, (-1, ), (0, ))).sum(1)
            padded, rolled_padded = torch.zeros((nfv.shape[0], 3), device=dev), torch.zeros((nfv.shape[0], 3), device=dev)
            padded[:, 0:2] = nfv
            rolled_padded[:, :2] = torch.roll(nfv, (-1, ), (0, ))
            sign = torch.sign(torch.cross(padded, rolled_padded, dim=1)[:, -1])

            angles.append(torch.acos(dot * (1 - 1e-6)) * sign)

        self.angles_fwd = angles[0]
        self.angles_bwd = angles[1]
        self.all_tfs = None
        if torch.isnan(self.angles_fwd).any() or torch.isinf(self.angles_fwd).any():
            a = 1

    def get_turning_functions(self, n, res):
        """
        Calculates the turning functions of this polygon starting at the nth vertex in both directions
        :param n: starting vertex id
        :param res: resolution of the function
        :return: "res" function values for both directions on the domain [0, 1]
        """
        x_fwd = torch.zeros((self.face_lengths_fwd.shape[0] + 1,))  # add lower domain limit (necessary for interpolation to work)
        x_bwd, y_fwd, y_bwd = x_fwd.clone(), x_fwd.clone(), x_fwd.clone()

        # get forward and backward direction x values of the turning function
        x_fwd[1:] = torch.cumsum(torch.roll(self.face_lengths_fwd, (-n,), (0,)), dim=0)
        x_bwd[1:] = torch.cumsum(torch.flip(torch.roll(self.face_lengths_fwd, (-n,), (0,)), dims=(0,)), dim=0)

        # same for the y-values
        y_fwd[:-1] = torch.cumsum(torch.roll(self.angles_fwd, (-n,), (0,)), dim=0)
        y_bwd[:-1] = torch.cumsum(torch.roll(self.angles_bwd, (-n,), (0,)), dim=0)
        y_fwd[-1] = y_fwd[-2]  # since we start at the lower domain limit with the first value we have not value at the upper limit
        y_bwd[-1] = y_bwd[-2]  # therefore duplicate the last value at the limit

        tfs_x = np.linspace(0, 1, res)  # this are all sample points of the turning function

        # get the y-values at all sample points (we cannot use pytorch here as there are no corresponding functions at least to my awareness)
        tfs_y_fwd = torch.from_numpy(interpolate.interp1d(x_fwd.cpu(), y_fwd.cpu(), kind='previous')(tfs_x)).to(self.poly_chain.device)
        tfs_y_bwd = torch.from_numpy(interpolate.interp1d(x_bwd.cpu(), y_bwd.cpu(), kind='previous')(tfs_x)).to(self.poly_chain.device)
        # return the sampled y-values
        ret = torch.stack([tfs_y_bwd, tfs_y_fwd])
        if torch.isnan(ret).any() or torch.isinf(ret).any():
            a = 1
        return ret


    def distance(self, other, res=100):
        """
        calculates the distance (similarity) of two Polygons by calculating the max value of the area that is enclosed
        by the two turning functions. Of the other polygon only one turning function is considered
        which is compared to all turning functions of this polygon
        :param other: other Polygon
        :param res: number of function values of the turning function (domain is [0, 1])
        :return: similarity score
        """

        # if not existing yet, get all possible turning functions of this object
        if self.all_tfs is None:
            self.all_tfs = torch.cat([self.get_turning_functions(n, res) for n in range(len(self.angles_fwd))], dim=0)

        # get one of all the possible turning functions of the other object
        other_tfs = other.get_turning_functions(0, res)[0][None]
        # calculate the distance between the other turning function and all of this turning functions by calculating
        # the area of the absolute enclosed area of every function pair. The minimum of all this values corresponds to
        # the most similar functions. This is what we want to return
        score = ((other_tfs - self.all_tfs) * (1/res)).abs().sum(1).min()
        score = score / (2 * np.pi)  # normalize by the largest possible area

        if torch.isnan(score).any() or torch.isinf(score).any():
            a = 1
        return score




