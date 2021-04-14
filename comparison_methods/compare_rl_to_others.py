import os
import torch
import elf
import numpy as np
import n_sphere
import wandb
from elf.segmentation.features import compute_rag
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import cm

from utils.general import cluster_embeddings, pca_project, random_label_cmap, multicut_from_probas, calculate_gt_edge_costs, get_angles
from utils.metrics import ClusterMetrics, SegmentationMetrics, AveragePrecision
from utils.affinities import get_edge_features_1d
from utils.yaml_conv_parser import dict_to_attrdict
from utils.training_helpers import update_env_data, Forwarder
from utils.distances import CosineDistance, L2Distance
from utils.affinities import get_affinities_from_embeddings_2d
from utils.graphs import get_soln_graph_clustering
from models.agent_model import Agent
from models.feature_extractor import FeExtractor
from environments.multicut import MulticutEmbeddingsEnv, State
from data.spg_dset import SpgDset



def validate_and_compare_to_clustering(model, env, distance, device, cfg):
    """validates the prediction against the method of clustering the embedding space"""

    model.eval()
    offs = [[1, 0], [0, 1], [2, 0], [0, 2], [4, 0], [0, 4], [16, 0], [0, 16]]
    ex_raws, ex_sps, ex_gts, ex_mc_gts, ex_embeds, ex_clst, ex_clst_sp, ex_mcaff, ex_mc_embed, ex_rl, \
    ex_clst_graph_agglo = [], [], [], [], [], [], [], [], [], [], []
    dset = SpgDset(cfg.val_data_dir, dict_to_attrdict(cfg.patch_manager), dict_to_attrdict(cfg.data_keys), max(cfg.s_subgraph))
    dloader = iter(DataLoader(dset))
    acc_reward = 0
    forwarder = Forwarder()
    delta_dist = 0.4

    segm_metric = AveragePrecision()
    clst_metric_rl = ClusterMetrics()
    clst_metric = ClusterMetrics()
    metric_sp_gt = ClusterMetrics()
    clst_metric_mcaff = ClusterMetrics()
    clst_metric_mcembed = ClusterMetrics()
    clst_metric_graphagglo = ClusterMetrics()

    map_rl, map_embed, map_sp_gt, map_mcaff, map_mcembed, map_graphagglo, map_rl, map_rl = [], [], [], [], [], [], [], []

    n_examples = len(dset)
    for it in range(n_examples):
        update_env_data(env, dloader, dset, device, with_gt_edges=False)
        env.reset()
        state = env.get_state()
        distr, _, _, _, _ = forwarder.forward(model, state, State, device, grad=False)
        action = torch.sigmoid(distr.loc)
        reward = env.execute_action(action, tau=0.0, train=False)
        acc_reward += reward[-2].item()

        embeddings = env.embeddings[0].cpu()
        node_features = env.current_node_embeddings.cpu().numpy()
        rag = env.rags[0]
        edge_ids = rag.uvIds()
        gt_seg = env.gt_seg[0].cpu().numpy()
        l2_embeddings = get_angles(embeddings[None])[0]
        l2_node_feats = get_angles(torch.from_numpy(node_features.T[None, ..., None])).squeeze().T.numpy()
        # clst_labels_kmeans = cluster_embeddings(l2_embeddings.permute((1, 2, 0)), len(np.unique(gt_seg)))
        # node_labels = cluster_embeddings(l2_node_feats, len(np.unique(gt_seg)))
        # clst_labels_sp_kmeans = elf.segmentation.features.project_node_labels_to_pixels(rag, node_labels).squeeze()

        clst_labels_sp_graph_agglo = get_soln_graph_clustering(env.init_sp_seg, torch.from_numpy(edge_ids.astype(np.int)), torch.from_numpy(l2_node_feats), len(np.unique(gt_seg)))[0][0].numpy()
        mc_labels_aff = env.get_current_soln(edge_weights=env.edge_features[:, 0]).cpu().numpy()[0]
        ew_embedaffs = 1 - get_edge_features_1d(env.init_sp_seg[0].cpu().numpy(), offs, get_affinities_from_embeddings_2d(env.embeddings, offs, delta_dist, distance)[0].cpu().numpy())[0][:, 0]
        mc_labels_embedding_aff = env.get_current_soln(edge_weights=torch.from_numpy(ew_embedaffs).to(device)).cpu().numpy()[0]
        rl_labels = env.current_soln.cpu().numpy()[0]

        ex_embeds.append(pca_project(embeddings, n_comps=3))
        ex_raws.append(env.raw[0].cpu().permute(1, 2, 0).squeeze())
        # ex_sps.append(cm.prism(env.init_sp_seg[0].cpu() / env.init_sp_seg[0].max().item()))
        ex_sps.append(env.init_sp_seg[0].cpu())
        gt_edges = calculate_gt_edge_costs(torch.from_numpy(edge_ids.astype(np.int)).to(device), env.init_sp_seg, torch.from_numpy(gt_seg).to(device), 0.3)
        ex_mc_gts.append(env.get_current_soln(edge_weights=gt_edges).cpu().numpy()[0])

        ex_gts.append(gt_seg)
        ex_rl.append(rl_labels)
        # ex_clst.append(clst_labels_kmeans)
        # ex_clst_sp.append(clst_labels_sp_kmeans)
        ex_clst_graph_agglo.append(clst_labels_sp_graph_agglo)
        ex_mcaff.append(mc_labels_aff)
        ex_mc_embed.append(mc_labels_embedding_aff)

        map_rl.append(segm_metric(rl_labels, gt_seg))
        clst_metric_rl(rl_labels, gt_seg)

        map_sp_gt.append(segm_metric(ex_mc_gts[-1], gt_seg))
        metric_sp_gt(ex_mc_gts[-1], gt_seg)

        # map_embed.append(segm_metric(clst_labels_kmeans, gt_seg))
        # clst_metric(clst_labels_kmeans, gt_seg)

        map_mcaff.append(segm_metric(mc_labels_aff, gt_seg))
        clst_metric_mcaff(mc_labels_aff, gt_seg)

        map_mcembed.append(segm_metric(mc_labels_embedding_aff, gt_seg))
        clst_metric_mcembed(mc_labels_embedding_aff, gt_seg)

        map_graphagglo.append(segm_metric(clst_labels_sp_graph_agglo, gt_seg))
        clst_metric_graphagglo(clst_labels_sp_graph_agglo.astype(np.int), gt_seg)

    print("\nmAP: ")
    print(f"sp gt       : {np.array(map_sp_gt).mean()}")
    print(f"ours        : {np.array(map_rl).mean()}")
    print(f"mc node     : {np.array(map_mcembed).mean()}")
    print(f"mc embed    : {np.array(map_mcaff).mean()}")
    print(f"graph agglo : {np.array(map_graphagglo).mean()}")

    vi_rl, are_rl, arp_rl, arr_rl = clst_metric_rl.dump()
    vi_spgt, are_spgt, arp_spgt, arr_spgt = metric_sp_gt.dump()
    vi_mcaff, are_mcaff, arp_mcaff, arr_mcaff = clst_metric_mcaff.dump()
    vi_mcembed, are_mcembed, arp_embed, arr_mcembed = clst_metric_mcembed.dump()
    vi_graphagglo, are_graphagglo, arp_graphagglo, arr_graphagglo = clst_metric_graphagglo.dump()

    print("\nVI: ")
    print(f"sp gt       : {vi_spgt}")
    print(f"ours        : {vi_rl}")
    print(f"mc affnties : {vi_mcaff}")
    print(f"mc embed    : {vi_mcembed}")
    print(f"graph agglo : {vi_graphagglo}")

    print("\nARE: ")
    print(f"sp gt       : {are_spgt}")
    print(f"ours        : {are_rl}")
    print(f"mc affnties : {are_mcaff}")
    print(f"mc embed    : {are_mcembed}")
    print(f"graph agglo : {are_graphagglo}")

    print("\nARP: ")
    print(f"sp gt       : {arp_spgt}")
    print(f"ours        : {arp_rl}")
    print(f"mc affnties : {arp_mcaff}")
    print(f"mc embed    : {arp_embed}")
    print(f"graph agglo : {arp_graphagglo}")

    print("\nARR: ")
    print(f"sp gt       : {arr_spgt}")
    print(f"ours        : {arr_rl}")
    print(f"mc affnties : {arr_mcaff}")
    print(f"mc embed    : {arr_mcembed}")
    print(f"graph agglo : {arr_graphagglo}")

    exit()
    for i in range(len(ex_gts)):
        fig, axs = plt.subplots(2, 4, figsize=(20, 13), sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
        axs[0, 0].imshow(ex_gts[i], cmap=random_label_cmap(), interpolation="none")
        axs[0, 0].set_title('gt')
        axs[0, 0].axis('off')
        axs[0, 1].imshow(ex_embeds[i])
        axs[0, 1].set_title('pc proj')
        axs[0, 1].axis('off')
        # axs[0, 2].imshow(ex_clst[i], cmap=random_label_cmap(), interpolation="none")
        # axs[0, 2].set_title('pix clst')
        # axs[0, 2].axis('off')
        axs[0, 2].imshow(ex_clst_graph_agglo[i], cmap=random_label_cmap(), interpolation="none")
        axs[0, 2].set_title('nagglo')
        axs[0, 2].axis('off')
        axs[0, 3].imshow(ex_mc_embed[i], cmap=random_label_cmap(), interpolation="none")
        axs[0, 3].set_title('mc embed')
        axs[0, 3].axis('off')
        axs[1, 0].imshow(ex_mc_gts[i], cmap=random_label_cmap(), interpolation="none")
        axs[1, 0].set_title('sp gt')
        axs[1, 0].axis('off')
        axs[1, 1].imshow(ex_sps[i], cmap=random_label_cmap(), interpolation="none")
        axs[1, 1].set_title('sp')
        axs[1, 1].axis('off')
        # axs[1, 2].imshow(ex_clst_sp[i], cmap=random_label_cmap(), interpolation="none")
        # axs[1, 2].set_title('sp clst')
        # axs[1, 2].axis('off')
        axs[1, 2].imshow(ex_rl[i], cmap=random_label_cmap(), interpolation="none")
        axs[1, 2].set_title('ours')
        axs[1, 2].axis('off')
        axs[1, 3].imshow(ex_mcaff[i], cmap=random_label_cmap(), interpolation="none")
        axs[1, 3].set_title('mc aff')
        axs[1, 3].axis('off')
        plt.show()
        # wandb.log({"validation/samples": [wandb.Image(fig, caption="sample images")]})
        plt.close('all')

if __name__=="__main__":
    wandb.init(project="dbg", entity="aule", config="../conf/leptin_configs.yaml")
    cfg = wandb.config
    device = torch.device("cuda:0")
    distance = CosineDistance()
    fe_ext = FeExtractor(dict_to_attrdict(cfg.backbone), distance, device)
    fe_ext.embed_model.load_state_dict(torch.load(cfg.fe_model_name))
    fe_ext.cuda(device)
    env = MulticutEmbeddingsEnv(fe_ext, cfg, device)
    model = Agent(cfg, State, distance, device)
    # model.load_state_dict(torch.load(cfg.agent_model_name))
    model.cuda(device)

    validate_and_compare_to_clustering(model, env, distance, device, cfg)