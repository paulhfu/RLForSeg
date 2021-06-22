import torch
from elf.segmentation.features import compute_rag


def update_env_data(env, data_iter, data_set, device, with_gt_edges=False, fe_grad=False):
    raw, gt, sp_seg, indices = next(data_iter)
    rags = [compute_rag(sseg.numpy()) for sseg in sp_seg]
    raw, gt, sp_seg = raw.to(device), gt.to(device), sp_seg.to(device)
    edges, gt_edges, edge_feat, node_feat = data_set.get_graphs(indices, sp_seg, device)
    # for e1, e2 in zip(edges, _edges):
    #     assert not (e1 != e2).any()

    env.update_data(edge_ids=edges, gt_edges=gt_edges, sp_seg=sp_seg, raw=raw, gt=gt, fe_grad=fe_grad, rags=rags,
                    edge_feat=edge_feat, node_feat=node_feat)


def state_to_cpu(state, state_class):
    state = list(state)
    for i in range(len(state)):
        if torch.is_tensor(state[i]):
            state[i] = state[i].cpu()
        elif isinstance(state[i], list) or isinstance(state[i], tuple):
            state[i] = state_to_cpu(state[i], None)
    if state_class is not None:
        return state_class(*state)
    return state


def state_to_cuda(state, device, state_class):
    state = list(state)
    for i in range(len(state)):
        if torch.is_tensor(state[i]):
            state[i] = state[i].to(device)
        elif isinstance(state[i], list) or isinstance(state[i], tuple):
            state[i] = state_to_cuda(state[i], device, None)
    if state_class is not None:
        return state_class(*state)
    return state

class Forwarder():
    def __init__(self):
        pass

    def forward(self, model, state, state_class, device, actions=None, grad=False, post_data=False, policy_opt=False,
                      get_node_feats=False, expl_action=None, get_embeddings=False):
        with torch.set_grad_enabled(grad):
            state = state_to_cuda(state, device, state_class)
            if actions is not None:
                actions = actions.to(model.device)
            ret = model(state,
                        actions,
                        expl_action,
                        post_data,
                        policy_opt and grad,
                        get_node_feats,
                        get_embeddings)
        return ret
