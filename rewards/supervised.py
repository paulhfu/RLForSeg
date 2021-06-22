import torch
import torch.nn as nn


class SubGraphDiceReward(object):

    def __init__(self, s_subgraph, *args, **kwargs):
        super(SubGraphDiceReward, self).__init__()
        self.epsilon = 1e-6
        self.class_weights = torch.tensor([1.0, 1.0]).unsqueeze(-1)
        self.s_subgraph = s_subgraph

    def __call__(self, actions, sg_gt_edges, subgraph_indices):
        reward = []
        sg_actions = []
        assert sg_gt_edges is not None, "did not get ground truth, therefore no dice score possible"
        for i, sz in enumerate(self.s_subgraph):
            sg_actions.append(
                actions[subgraph_indices[i].view(-1, sz)])
        for i in range(len(sg_actions)):
            input = torch.stack([1-sg_actions[i], sg_actions[i]], 0)
            target = torch.stack([1-sg_gt_edges[i], sg_gt_edges[i]], 0).float()
            intersect = (input * target).sum(-1)

            denominator = (input * input).sum(-1) + (target * target).sum(-1)
            dice_score = 2 * (intersect / denominator.clamp(min=self.epsilon))
            dice_score = dice_score * self.class_weights.to(dice_score.device)

            reward.append(dice_score.sum(0)*0.5)
        return reward, torch.cat(dice_score, 0).mean()
