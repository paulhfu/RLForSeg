import os
import torch
import elf
import numpy as np
import wandb
from elf.segmentation.features import compute_rag
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import cm

from utils.general import cluster_embeddings, pca_project, random_label_cmap
from utils.matching import matching
from utils.yaml_conv_parser import dict_to_attrdict
from utils.training_helpers import update_env_data, agent_forward
from utils.distances import CosineDistance, L2Distance
from models.agent_model import Agent
from models.feature_extractor import FeExtractor
from environments.multicut import MulticutEmbeddingsEnv
from data.spg_dset import SpgDset



def validate_and_compare_to_clustering(self, model, env, dset, device):
    """validates the prediction against the method of clustering the embedding space"""

    if self.cfg.verbose:
        print("\n\n###### start validate ######", end='')
    model.eval()
    n_examples = 1
    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    clst_scores, clst_scores_sp, rl_scores, keys = [], [], [], None
    ex_raws, ex_sps, ex_gts, ex_mc_gts, ex_embeds, ex_clst, ex_clst_sp, ex_rl = [], [], [], [], [], [], [], []
    dloader = DataLoader(dset, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
    acc_reward = 0
    for it in range(len(dset)):
        update_env_data(env, dloader, self.cfg, device, with_gt_edges="sub_graph_dice" in self.cfg.reward_function)
        env.reset()
        state = env.get_state()

        distr, _, _, _, _, _ = agent_forward(env, model, state, grad=False, post_data=False)
        action = torch.sigmoid(distr.loc)
        reward = env.execute_action(action, None, post_images=False, tau=0.0, train=False)
        acc_reward += reward[-1].item()
        if self.cfg.verbose:
            print(
                f"\nstep: {it}; mean_loc: {round(distr.loc.mean().item(), 5)}; mean reward: {round(reward[-1].item(), 5)}",
                end='')

        embeddings = env.embeddings[0].cpu().numpy()
        node_features = env.current_node_embeddings.cpu().numpy()
        rag = env.rags[0]
        gt_seg = env.gt_seg[0].cpu().numpy()
        gt_mc = cm.prism(
            env.gt_soln[0].cpu() / env.gt_soln[0].max().item()) if env.gt_edge_weights is not None else torch.zeros(
            env.raw.shape[-2:])
        clst_labels = cluster_embeddings(embeddings.transpose((1, 2, 0)), len(np.unique(gt_seg)))
        node_labels = cluster_embeddings(node_features, len(np.unique(gt_seg)))
        clst_labels_sp = elf.segmentation.features.project_node_labels_to_pixels(rag, node_labels).squeeze()
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

        _clst_scores = matching(gt_seg, clst_labels, thresh=taus, criterion='iou', report_matches=False)
        _clst_scores_sp = matching(gt_seg, clst_labels_sp, thresh=taus, criterion='iou', report_matches=False)
        _rl_scores = matching(gt_seg, rl_labels, thresh=taus, criterion='iou', report_matches=False)

        if it == 0:
            for tau_it in range(len(_clst_scores)):
                clst_scores.append(np.array(list(map(float, list(_clst_scores[tau_it]._asdict().values())[1:]))))
                clst_scores_sp.append(
                    np.array(list(map(float, list(_clst_scores_sp[tau_it]._asdict().values())[1:]))))
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
    plt.subplots_adjust(hspace=.5)

    for m in ('precision', 'recall', 'accuracy', 'f1'):
        axs[0, 0].plot(taus, [s[m] for s in rl_scores], '.-', lw=2, label=m)
    axs[0, 0].set_ylabel('Metric value')
    axs[0, 0].grid()
    axs[0, 0].legend(bbox_to_anchor=(.8, 1.65), loc='upper left', fontsize='xx-small')
    axs[0, 0].set_title('RL method')

    for m in ('fp', 'tp', 'fn'):
        axs[0, 1].plot(taus, [s[m] for s in rl_scores], '.-', lw=2, label=m)
    axs[0, 1].set_ylabel('Number #')
    axs[0, 1].grid()
    axs[0, 1].legend(bbox_to_anchor=(.87, 1.6), loc='upper left', fontsize='xx-small');
    axs[0, 1].set_title('RL method')

    for m in ('precision', 'recall', 'accuracy', 'f1'):
        axs[1, 0].plot(taus, [s[m] for s in clst_scores], '.-', lw=2, label=m)
    axs[1, 0].set_ylabel('Metric value')
    axs[1, 0].grid()
    axs[1, 0].set_title('Clustering pix')

    for m in ('fp', 'tp', 'fn'):
        axs[1, 1].plot(taus, [s[m] for s in clst_scores], '.-', lw=2, label=m)
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

    wandb.log({"validation/metrics": [wandb.Image(fig, caption="metrics")]})
    wandb.log({"validation/reward": acc_reward})
    plt.close('all')
    if acc_reward > self.best_val_reward:
        self.best_val_reward = acc_reward
        wandb.run.summary["validation/reward"] = acc_reward
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, "best_checkpoint_agent.pth"))
    if self.cfg.verbose:
        print("\n###### finish validate ######\n", end='')

    for i in range(n_examples):
        fig, axs = plt.subplots(2, 4, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
        axs[0, 0].imshow(ex_gts[i], cmap=random_label_cmap(), interpolation="none")
        axs[0, 0].set_title('gt')
        axs[0, 0].axis('off')
        axs[0, 1].imshow(ex_raws[i])
        axs[0, 1].set_title('raw image')
        axs[0, 1].axis('off')
        axs[0, 2].imshow(ex_sps[i], cmap=random_label_cmap(), interpolation="none")
        axs[0, 2].set_title('superpixels')
        axs[0, 2].axis('off')
        axs[0, 3].imshow(ex_embeds[i])
        axs[0, 3].set_title('pc proj 1-3')
        axs[0, 3].axis('off')
        axs[1, 0].imshow(ex_mc_gts[i], cmap=random_label_cmap(), interpolation="none")
        axs[1, 0].set_title('sp gt')
        axs[1, 0].axis('off')
        axs[1, 1].imshow(ex_clst[i], cmap=random_label_cmap(), interpolation="none")
        axs[1, 1].set_title('pred pix clst')
        axs[1, 1].axis('off')
        axs[1, 2].imshow(ex_clst_sp[i], cmap=random_label_cmap(), interpolation="none")
        axs[1, 2].set_title('pred sp clst')
        axs[1, 2].axis('off')
        axs[1, 3].imshow(ex_rl[i], cmap=random_label_cmap(), interpolation="none")
        axs[1, 3].set_title('pred RL')
        axs[1, 3].axis('off')
        wandb.log({"validation/samples": [wandb.Image(fig, caption="sample images")]})
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
    model.load_state_dict(torch.load(cfg.agent_model_name))
    model.cuda(device)

    validate_and_compare_to_clustering(model, env, dset, device)