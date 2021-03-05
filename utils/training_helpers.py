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
    if not all([e.shape[-1] > cfg.s_subgraph[-1] for e in edges]):
        print("ERROR not enough edges, subgraph generation will fail")
        assert False
    if with_gt_edges:
        gt_edges = [torch.from_numpy(
            calculate_gt_edge_costs(s_edges.T.cpu().numpy(), sseg.squeeze().numpy(), sgt.squeeze().numpy(),
                                    cfg.gt_edge_overlap_thresh)).to(device).float() for s_edges, sseg, sgt in
                    zip(edges, sp_seg, gt)]
    else:
        gt_edges = None
    raw, gt, sp_seg = raw.to(device), gt.to(device), sp_seg.to(device)
    env.update_data(edge_ids=edges, gt_edges=gt_edges, sp_seg=sp_seg, raw=raw, gt=gt, fe_grad=fe_grad, rags=rags)


def validate(model, env, cfg, device):
    """validates the prediction against the method of clustering the embedding space"""

    model.eval()
    n_examples = 4
    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    dset = SpgDset(cfg.data_dir, cfg.patch_manager, max(cfg.s_subgraph))
    dloader = DataLoader(dset, batch_size=1, shuffle=True, pin_memory=True,
                         num_workers=0)
    clst_scores, clst_scores_sp, rl_scores, keys = [], [], [], None
    ex_raws, ex_sps, ex_gts, ex_embeds, ex_clst, ex_clst_sp, ex_rl = [], [], [], [], [], [], []
    for it in range(10):
        update_env_data(env, dloader, device)
        env.reset()
        state = env.get_state()
        _, _, _, action, embeddings, _, node_features = agent_forward(env, model, state=state, grad=False,
                                                                      post_input=False, post_model=False,
                                                                      return_node_features=True)
        env.execute_action(action, None, post_stats=False)

        clst_labels = cluster_embeddings(embeddings.cpu().numpy().squeeze().transpose((1, 2, 0)),
                                         len(torch.unique(env.gt_seg)))
        node_labels = cluster_embeddings(node_features.cpu().numpy().squeeze(), len(torch.unique(env.gt_seg)))
        rag = elf.segmentation.features.compute_rag(np.expand_dims(env.init_sp_seg.cpu().numpy().squeeze(), axis=0))
        clst_labels_sp = elf.segmentation.features.project_node_labels_to_pixels(rag, node_labels).squeeze()
        rl_labels = env.current_soln.cpu().numpy().squeeze()

        if it < n_examples:
            ex_embeds.append(pca_project(embeddings.detach().squeeze().cpu().numpy(), n_comps=6))
            ex_raws.append(env.raw[0].cpu().permute(1, 2, 0).squeeze())
            ex_sps.append(cm.prism(env.init_sp_seg[0].cpu() / env.init_sp_seg[0].max().item()))
            ex_gts.append(cm.prism(env.gt_soln[0].cpu() / env.gt_soln[0].max().item()))
            ex_clst.append(cm.prism(clst_labels / clst_labels.max()))
            ex_clst_sp.append(cm.prism(clst_labels_sp / clst_labels_sp.max()))
            ex_rl.append(cm.prism(rl_labels / rl_labels.max()))

        _clst_scores = matching(env.gt_seg.cpu().numpy().squeeze(), clst_labels.squeeze(),
                                thresh=taus, criterion='iou', report_matches=False)

        _clst_scores_sp = matching(env.gt_seg.cpu().numpy().squeeze(), clst_labels_sp.squeeze(),
                                   thresh=taus, criterion='iou', report_matches=False)

        _rl_scores = matching(env.gt_seg.cpu().numpy().squeeze(), rl_labels,
                              thresh=taus, criterion='iou', report_matches=False)

        if it == 0:
            for tau_it in range(len(_clst_scores)):
                clst_scores.append(np.array(list(map(float, list(_clst_scores[tau_it]._asdict().values())[1:]))))
                clst_scores_sp.append(np.array(list(map(float, list(_clst_scores_sp[tau_it]._asdict().values())[1:]))))
                rl_scores.append(np.array(list(map(float, list(_rl_scores[tau_it]._asdict().values())[1:]))))
            keys = list(_clst_scores[0]._asdict().keys())[1:]
        else:
            for tau_it in range(len(_clst_scores)):
                clst_scores[tau_it] += np.array(list(map(float, list(_clst_scores[tau_it]._asdict().values())[1:])))
                clst_scores_sp[tau_it] += np.array(
                    list(map(float, list(_clst_scores_sp[tau_it]._asdict().values())[1:])))
                rl_scores[tau_it] += np.array(list(map(float, list(_rl_scores[tau_it]._asdict().values())[1:])))

    div = np.ones_like(clst_scores[0])
    for i, key in enumerate(keys):
        if key not in ('fp', 'tp', 'fn'):
            div[i] = 10

    for tau_it in range(len(clst_scores)):
        clst_scores[tau_it] = dict(zip(keys, clst_scores[tau_it] / div))
        clst_scores_sp[tau_it] = dict(zip(keys, clst_scores_sp[tau_it] / div))
        rl_scores[tau_it] = dict(zip(keys, rl_scores[tau_it] / div))

    fig, axs = plt.subplots(3, 2, figsize=(10, 5))

    for m in ('precision', 'recall', 'accuracy', 'f1'):
        axs[0, 0].plot(taus, [s[m] for s in rl_scores], '.-', lw=2, label=m)
    axs[0, 0].set_xlabel(r'IoU threshold $\tau$')
    axs[0, 0].set_ylabel('Metric value')
    axs[0, 0].grid()
    axs[0, 0].legend()
    axs[0, 0].set_title('RL method')

    for m in ('fp', 'tp', 'fn'):
        axs[0, 1].plot(taus, [s[m] for s in rl_scores], '.-', lw=2, label=m)
    axs[0, 1].set_xlabel(r'IoU threshold $\tau$')
    axs[0, 1].set_ylabel('Number #')
    axs[0, 1].grid()
    axs[0, 1].legend();
    axs[0, 1].set_title('RL method')

    for m in ('precision', 'recall', 'accuracy', 'f1'):
        axs[1, 0].plot(taus, [s[m] for s in clst_scores], '.-', lw=2, label=m)
    axs[1, 0].set_xlabel(r'IoU threshold $\tau$')
    axs[1, 0].set_ylabel('Metric value')
    axs[1, 0].grid()
    axs[1, 0].set_title('Clustering pix')

    for m in ('fp', 'tp', 'fn'):
        axs[1, 1].plot(taus, [s[m] for s in clst_scores], '.-', lw=2, label=m)
    axs[1, 1].set_xlabel(r'IoU threshold $\tau$')
    axs[1, 1].set_ylabel('Number #')
    axs[1, 1].grid()
    axs[1, 1].set_title('Clustering pix')

    for m in ('precision', 'recall', 'accuracy', 'f1'):
        axs[2, 0].plot(taus, [s[m] for s in clst_scores_sp], '.-', lw=2, label=m)
    axs[2, 0].set_xlabel(r'IoU threshold $\tau$')
    axs[2, 0].set_ylabel('Metric value')
    axs[2, 0].grid()
    axs[2, 0].set_title('Clustering sp')

    for m in ('fp', 'tp', 'fn'):
        axs[2, 1].plot(taus, [s[m] for s in clst_scores_sp], '.-', lw=2, label=m)
    axs[2, 1].set_xlabel(r'IoU threshold $\tau$')
    axs[2, 1].set_ylabel('Number #')
    axs[2, 1].grid()
    axs[2, 1].set_title('Clustering sp')

    plt.show()

    for i in range(n_examples):
        fig, axs = plt.subplots(2, 4, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
        axs[0, 0].imshow(ex_embeds[i][..., :3])
        axs[0, 0].set_title('pc proj 1-3')
        axs[0, 1].imshow(ex_raws[i])
        axs[0, 1].set_title('raw image')
        axs[0, 2].imshow(ex_sps[i])
        axs[0, 2].set_title('superpixels')
        axs[0, 3].imshow(ex_gts[i])
        axs[0, 3].set_title('gt')

        axs[1, 0].imshow(ex_embeds[i][..., 3:])
        axs[1, 0].set_title('pc proj 3-6', y=-0.3)
        axs[1, 1].imshow(ex_clst[i])
        axs[1, 1].set_title('pred pix clst', y=-0.3)
        axs[1, 2].imshow(ex_clst_sp[i])
        axs[1, 2].set_title('pred sp clst', y=-0.3)
        axs[1, 3].imshow(ex_rl[i])
        axs[1, 3].set_title('pred RL', y=-0.3)
        plt.show()

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
