from abc import ABCMeta, abstractmethod
import torch

class RewardFunctionAbc(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, shape_samples: torch.Tensor):
        pass

    @abstractmethod
    def __call__(self, prediction_segmentation: torch.Tensor, superpixel_segmentation: torch.Tensor):
        """
        This method should give a score for each label in superpixel_segmentation based on the objects in
        prediction_segmentation and prior knowledge. This scoring can be roughly sketched by the
        following:
            - Find out background and foreground objects in prediction_segmentation.
            - For the foreground objects their respective shape should be obtained and compared with prior.
              This produces a score for each of the foreground objects.
            - Find the superpixels that compose each object and assign the objects score to its superpixels.
            - Do some global scoring. E.g. if there are too few objects give negative scores to the superpixels
              within the background.

        :param prediction_segmentation: tensor of shape N|H|W
        :param superpixel_segmentation: tensor of shape N|H|W
        :return: list of torch.Tensor where each entry in a tensor represents the reward scora for a subgraph,
                  global reward score
        """
        pass

