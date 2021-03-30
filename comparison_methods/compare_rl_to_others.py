import os
import torch
import elf
import numpy as np
import wandb
from elf.segmentation.features import compute_rag
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import cm

from utils.general import cluster_embeddings, pca_project, random_label_cmap, multicut_from_probas
from utils.matching import matching
from utils.affinities import get_edge_features_1d
from utils.yaml_conv_parser import dict_to_attrdict
from utils.training_helpers import update_env_data, agent_forward
from utils.distances import CosineDistance, L2Distance
from utils.affinities import get_affinities_from_embeddings_2d
from models.agent_model import Agent
from models.feature_extractor import FeExtractor
from environments.multicut import MulticutEmbeddingsEnv
from data.spg_dset import SpgDset



def validate_and_compare_to_clustering(model, env, dset, device, cfg):
    """validates the prediction against the method of clustering the embedding space"""

    model.eval()
    n_examples = 100
    offs = [[1, 0], [0, 1], [2, 0], [0, 2], [4, 0], [0, 4], [16, 0], [0, 16]]
    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    clst_scores, clst_scores_sp, mc_scores_unsup_aff, mc_scores_embedding_aff, rl_scores= [], [], [], [], []
    keys = None
    ex_raws, ex_sps, ex_gts, ex_mc_gts, ex_embeds, ex_clst, ex_clst_sp, ex_mcaff, ex_mc_embed, ex_rl = [], [], [], [], [], [], [], [], [], []
    dloader = DataLoader(dset, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
    acc_reward = 0
    for it in range(len(dset)):
        update_env_data(env, dloader, cfg, device, with_gt_edges="sub_graph_dice" in cfg.reward_function)
        env.reset()
        state = env.get_state()

        distr, _, _, _, _, _ = agent_forward(env, model, state, grad=False, post_data=False)
        action = torch.sigmoid(distr.loc)
        reward = env.execute_action(action, None, post_images=False, tau=0.0, train=False)
        acc_reward += reward[-1].item()
        if cfg.verbose:
            print(
                f"\nstep: {it}; mean_loc: {round(distr.loc.mean().item(), 5)}; mean reward: {round(reward[-1].item(), 5)}",
                end='')

        embeddings = env.embeddings[0].cpu().numpy()
        node_features = env.current_node_embeddings.cpu().numpy()
        rag = env.rags[0]
        gt_seg = env.noisy_gt_seg[0].cpu().numpy()
        gt_mc = cm.prism(
            env.gt_soln[0].cpu() / env.gt_soln[0].max().item()) if env.gt_edge_weights is not None else torch.zeros(
            env.raw.shape[-2:])
        clst_labels = cluster_embeddings(embeddings.transpose((1, 2, 0)), len(np.unique(gt_seg)))
        node_labels = cluster_embeddings(node_features, len(np.unique(gt_seg)))
        clst_labels_sp = elf.segmentation.features.project_node_labels_to_pixels(rag, node_labels).squeeze()
        mc_labels_unsup_aff = env.get_current_soln(edge_weights=env.edge_feats[:, 0]).cpu().numpy()[0]
        ew_embedaffs = 1 - get_edge_features_1d(env.init_sp_seg[0].cpu().numpy(), offs, get_affinities_from_embeddings_2d(env.embeddings, offs, cfg.delta_dist, distance)[0].cpu().numpy())[0][:, 0]
        mc_labels_embedding_aff = env.get_current_soln(edge_weights=torch.from_numpy(ew_embedaffs).to(device)).cpu().numpy()[0]
        rl_labels = env.current_soln.cpu().numpy()[0]

        if it < n_examples:
            ex_embeds.append(pca_project(embeddings, n_comps=3))
            ex_raws.append(env.raw[0].cpu().permute(1, 2, 0).squeeze())
            # ex_sps.append(cm.prism(env.init_sp_seg[0].cpu() / env.init_sp_seg[0].max().item()))
            ex_sps.append(env.init_sp_seg[0].cpu())
            ex_mc_gts.append(gt_mc)
            ex_gts.append(gt_seg)
            ex_clst.append(clst_labels)
            ex_clst_sp.append(clst_labels_sp)
            ex_rl.append(rl_labels)
            ex_mcaff.append(mc_labels_unsup_aff)
            ex_mc_embed.append(mc_labels_embedding_aff)

        _clst_scores = matching(gt_seg, clst_labels, thresh=taus, criterion='iou', report_matches=False)
        _clst_scores_sp = matching(gt_seg, clst_labels_sp, thresh=taus, criterion='iou', report_matches=False)
        _rl_scores = matching(gt_seg, rl_labels, thresh=taus, criterion='iou', report_matches=False)
        _mc_scores_unsup_aff = matching(gt_seg, mc_labels_unsup_aff, thresh=taus, criterion='iou', report_matches=False)
        _mc_scores_embedding_aff = matching(gt_seg, mc_labels_embedding_aff, thresh=taus, criterion='iou', report_matches=False)

        if it == 0:
            for tau_it in range(len(_clst_scores)):
                clst_scores.append(np.array(list(map(float, list(_clst_scores[tau_it]._asdict().values())[1:]))))
                clst_scores_sp.append(np.array(list(map(float, list(_clst_scores_sp[tau_it]._asdict().values())[1:]))))
                rl_scores.append(np.array(list(map(float, list(_rl_scores[tau_it]._asdict().values())[1:]))))
                mc_scores_unsup_aff.append(np.array(list(map(float, list(_mc_scores_unsup_aff[tau_it]._asdict().values())[1:]))))
                mc_scores_embedding_aff.append(np.array(list(map(float, list(_mc_scores_embedding_aff[tau_it]._asdict().values())[1:]))))
            keys = list(_clst_scores[0]._asdict().keys())[1:]
        else:
            for tau_it in range(len(_clst_scores)):
                clst_scores[tau_it] += np.array(list(map(float, list(_clst_scores[tau_it]._asdict().values())[1:])))
                clst_scores_sp[tau_it] += np.array(list(map(float, list(_clst_scores_sp[tau_it]._asdict().values())[1:])))
                rl_scores[tau_it] += np.array(list(map(float, list(_rl_scores[tau_it]._asdict().values())[1:])))
                mc_scores_unsup_aff[tau_it] += np.array(list(map(float, list(_mc_scores_unsup_aff[tau_it]._asdict().values())[1:])))
                mc_scores_embedding_aff[tau_it] += np.array(list(map(float, list(_mc_scores_embedding_aff[tau_it]._asdict().values())[1:])))

    div = np.ones_like(clst_scores[0])
    for i, key in enumerate(keys):
        if key not in ('fp', 'tp', 'fn'):
            div[i] = 10

    for tau_it in range(len(clst_scores)):
        clst_scores[tau_it] = dict(zip(keys, clst_scores[tau_it] / div))
        clst_scores_sp[tau_it] = dict(zip(keys, clst_scores_sp[tau_it] / div))
        rl_scores[tau_it] = dict(zip(keys, rl_scores[tau_it] / div))
        mc_scores_unsup_aff[tau_it] = dict(zip(keys, mc_scores_unsup_aff[tau_it] / div))
        mc_scores_embedding_aff[tau_it] = dict(zip(keys, mc_scores_embedding_aff[tau_it] / div))

    fig, axs = plt.subplots(5, 2, figsize=(10, 5))
    plt.subplots_adjust(hspace=1.0)

    for m in ('precision', 'recall', 'accuracy', 'f1'):
        axs[0, 0].plot(taus, [s[m] for s in rl_scores], '.-', lw=2, label=m)
    axs[0, 0].set_ylabel('Metric value')
    axs[0, 0].grid()
    axs[0, 0].legend(bbox_to_anchor=(.8, 1.65), loc='upper left', fontsize='xx-small')
    axs[0, 0].set_title('ours')

    for m in ('fp', 'tp', 'fn'):
        axs[0, 1].plot(taus, [s[m] for s in rl_scores], '.-', lw=2, label=m)
    axs[0, 1].set_ylabel('Number')
    axs[0, 1].grid()
    axs[0, 1].legend(bbox_to_anchor=(.87, 1.6), loc='upper left', fontsize='xx-small');
    axs[0, 1].set_title('ours')

    for m in ('precision', 'recall', 'accuracy', 'f1'):
        axs[1, 0].plot(taus, [s[m] for s in clst_scores], '.-', lw=2, label=m)
    axs[1, 0].grid()
    axs[1, 0].set_title('pix clst')

    for m in ('fp', 'tp', 'fn'):
        axs[1, 1].plot(taus, [s[m] for s in clst_scores], '.-', lw=2, label=m)
    axs[1, 1].grid()
    axs[1, 1].set_title('pix clst')

    for m in ('precision', 'recall', 'accuracy', 'f1'):
        axs[2, 0].plot(taus, [s[m] for s in clst_scores_sp], '.-', lw=2, label=m)
    axs[2, 0].grid()
    axs[2, 0].set_title('sp clst')

    for m in ('fp', 'tp', 'fn'):
        axs[2, 1].plot(taus, [s[m] for s in clst_scores_sp], '.-', lw=2, label=m)
    axs[2, 1].grid()
    axs[2, 1].set_title('sp clst')

    for m in ('precision', 'recall', 'accuracy', 'f1'):
        axs[3, 0].plot(taus, [s[m] for s in mc_scores_unsup_aff], '.-', lw=2, label=m)
    axs[3, 0].grid()
    axs[3, 0].set_title('mc raw')

    for m in ('fp', 'tp', 'fn'):
        axs[3, 1].plot(taus, [s[m] for s in mc_scores_unsup_aff], '.-', lw=2, label=m)
    axs[3, 1].grid()
    axs[3, 1].set_title('mc raw')

    for m in ('precision', 'recall', 'accuracy', 'f1'):
        axs[4, 0].plot(taus, [s[m] for s in mc_scores_embedding_aff], '.-', lw=2, label=m)
    axs[4, 0].set_xlabel(r'IoU threshold $\tau$')
    axs[4, 0].grid()
    axs[4, 0].set_title('mc embed')

    for m in ('fp', 'tp', 'fn'):
        axs[4, 1].plot(taus, [s[m] for s in mc_scores_embedding_aff], '.-', lw=2, label=m)
    axs[4, 1].set_xlabel(r'IoU threshold $\tau$')
    axs[4, 1].grid()
    axs[4, 1].set_title('mc embed')

    plt.show()
    # wandb.log({"validation/metrics": [wandb.Image(fig, caption="metrics")]})
    # wandb.log({"validation/reward_RL": acc_reward})
    plt.close('all')

    for i in range(len(ex_gts)):
        fig, axs = plt.subplots(2, 5, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
        axs[0, 0].imshow(ex_gts[i], cmap=random_label_cmap(), interpolation="none")
        axs[0, 0].set_title('gt')
        axs[0, 0].axis('off')
        axs[0, 1].imshow(ex_raws[i])
        axs[0, 1].set_title('raw image')
        axs[0, 1].axis('off')
        axs[0, 2].imshow(ex_sps[i], cmap=random_label_cmap(), interpolation="none")
        axs[0, 2].set_title('sp')
        axs[0, 2].axis('off')
        axs[0, 3].imshow(ex_embeds[i])
        axs[0, 3].set_title('embed proj')
        axs[0, 3].axis('off')
        axs[1, 0].imshow(ex_mc_gts[i], cmap=random_label_cmap(), interpolation="none")
        axs[1, 0].set_title('sp gt')
        axs[1, 0].axis('off')
        axs[1, 1].imshow(ex_clst[i], cmap=random_label_cmap(), interpolation="none")
        axs[1, 1].set_title('pix clst')
        axs[1, 1].axis('off')
        axs[1, 2].imshow(ex_clst_sp[i], cmap=random_label_cmap(), interpolation="none")
        axs[1, 2].set_title('sp clst')
        axs[1, 2].axis('off')
        axs[1, 3].imshow(ex_rl[i], cmap=random_label_cmap(), interpolation="none")
        axs[1, 3].set_title('ours')
        axs[1, 3].axis('off')
        axs[0, 4].imshow(ex_mc_embed[i], cmap=random_label_cmap(), interpolation="none")
        axs[0, 4].set_title('mc embed')
        axs[0, 4].axis('off')
        axs[1, 4].imshow(ex_mcaff[i], cmap=random_label_cmap(), interpolation="none")
        axs[1, 4].set_title('mc raw')
        axs[1, 4].axis('off')
        plt.show()
        # wandb.log({"validation/samples": [wandb.Image(fig, caption="sample images")]})
        plt.close('all')

if __name__=="__main__":
    wandb.init(project="Compare RL to others", entity="aule", config="./default_configs.yaml")
    cfg = wandb.config
    device = torch.device("cuda:0")
    distance = CosineDistance()
    dset = SpgDset(cfg.data_dir, dict_to_attrdict(cfg.patch_manager), max(cfg.s_subgraph))
    fe_ext = FeExtractor(dict_to_attrdict(cfg.backbone), distance, device)
    fe_ext.load_state_dict(torch.load(cfg.fe_model_name))
    fe_ext.cuda(device)
    env = MulticutEmbeddingsEnv(fe_ext, cfg, device)
    model = Agent(cfg, env.State, distance, device)
    # model.load_state_dict(torch.load(cfg.agent_model_name))
    model.cuda(device)

    validate_and_compare_to_clustering(model, env, dset, device, cfg)