import os
import torch
import elf
import copy
import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from elf.segmentation.features import compute_rag
from data.spg_dset import SpgDset
from torch.utils.data import DataLoader
from utils.general import get_angles, adjust_learning_rate, cluster_embeddings, pca_project, calculate_gt_edge_costs
from utils.matching import matching
from utils.yaml_conv_parser import AttrDict, add_dict
from torch.optim.lr_scheduler import ReduceLROnPlateau


def update_env_data(env, dloader, cfg, device, with_gt_edges=False, fe_grad=False):
    raw, gt, sp_seg, indices = next(iter(dloader))
    rags = [compute_rag(sseg.numpy()) for sseg in sp_seg]
    edges = [torch.from_numpy(rag.uvIds().astype(np.long)).T.to(device) for rag in rags]
    raw, gt, sp_seg = raw.to(device), gt.to(device), sp_seg.to(device)
    if not all([e.shape[-1] > cfg.s_subgraph[-1] for e in edges]):
        print("ERROR not enough edges, subgraph generation will fail")
        assert False
    if with_gt_edges:
        _edges, edge_feat, _, gt_edges = dloader.dataset.get_graphs(indices, sp_seg, device)
        for e1, e2 in zip(edges, _edges):
            assert not (e1 != e2).any()
        # gt_edges = [calculate_gt_edge_costs(s_edges.T, sseg.squeeze(), sgt.squeeze(), cfg.gt_edge_overlap_thresh).to(device).float() for s_edges, sseg, sgt in zip(edges, sp_seg, gt)]
    else:
        gt_edges = None
    env.update_data(edge_ids=edges, gt_edges=gt_edges, sp_seg=sp_seg, raw=raw, gt=gt, fe_grad=fe_grad, rags=rags, edge_features=edge_feat)


def supervised_policy_pretraining(model, env, cfg, device="cuda:0", fe_opt=False):
    wu_cfg = AttrDict()
    add_dict(cfg.policy_warmup, wu_cfg)
    dset = SpgDset(cfg.data_dir, wu_cfg.patch_manager, max(cfg.trn.s_subgraph))
    dloader = DataLoader(dset, batch_size=wu_cfg.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    if fe_opt:
        actor_fe_opt = torch.optim.Adam(list(model.actor.parameters()) + list(env.embedding_net.parameters()),
                                        lr=wu_cfg.lr)
    else:
        actor_fe_opt = torch.optim.Adam(model.actor.parameters(), lr=wu_cfg.lr)

    dummy_opt = torch.optim.Adam([model.log_alpha], lr=wu_cfg.lr)
    sheduler = ReduceLROnPlateau(actor_fe_opt, threshold=0.001, min_lr=1e-6)
    criterion = torch.nn.BCELoss()
    acc_loss = 0
    iteration = 0
    best_score = -np.inf
    # be careful with this, it assumes a one step episode environment
    while iteration <= wu_cfg.n_iterations:
        update_env_data(env, dloader, device, with_gt_edges=True, fe_grad=fe_opt)
        state = env.get_state()
        # Calculate policy and values
        distr, q1, q2, _, _ = agent_forward(env, model, state, policy_opt=True)
        action = distr.transforms[0](distr.loc)
        loss = criterion(action.squeeze(1), env.gt_edge_weights)

        dummy_loss = (
                model.alpha * 0).sum()  # not using all parameters in backprop gives error, so add dummy loss
        for sq1, sq2 in zip(q1, q2):
            loss = loss + (sq1.sum() * sq2.sum() * 0)

        actor_fe_opt.zero_grad()
        loss.backward(retain_graph=False)
        actor_fe_opt.step()

        dummy_opt.zero_grad()
        dummy_loss.backward(retain_graph=False)
        dummy_opt.step()
        acc_loss += loss.item()

        if iteration % 10 == 0:
            _, reward = env.execute_action(action.detach(), None, post_images=True, tau=0.0)
            sheduler.step(acc_loss / 10)
            total_reward = 0
            for _rew in reward:
                total_reward += _rew.mean().item()
            total_reward /= len(reward)
            wandb.log({"policy_warm_start/acc_loss": acc_loss})
            wandb.log({"policy_warm_start/rewards": total_reward})
            acc_loss = 0
            if total_reward > best_score:
                best_model = copy.deepcopy(model.state_dict())
                best_score = total_reward
        wandb.log({"policy_warm_start/loss": loss.item()})
        wandb.log({"policy_warm_start/lr": actor_fe_opt.param_groups[0]['lr']})
        iteration += 1
    model.load_state_dict(best_model)
    return


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


def agent_forward(env, model, state, actions=None, grad=True, post_data=False, policy_opt=False,
                  return_node_features=False):
    with torch.set_grad_enabled(grad):
        state = state_to_cuda(state, env.device, env.State)
        if actions is not None:
            actions = actions.to(model.device)
        ret = model(state,
                    actions,
                    post_data,
                    policy_opt and grad,
                    return_node_features)
    return ret
