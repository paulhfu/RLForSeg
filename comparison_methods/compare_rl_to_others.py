import os
import torch
import elf
import numpy as np
import n_sphere
import wandb
import z5py
from elf.segmentation.features import compute_rag
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import cm

from utils.general import cluster_embeddings, pca_project, random_label_cmap, multicut_from_probas, project_overseg_to_seg, get_angles
from utils.metrics import ClusterMetrics, SegmentationMetrics, AveragePrecision, SBD
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
    ex_clst_graph_agglo= [], [], [], [], [], [], [], [], [], [], []
    dset = SpgDset(cfg.val_data_dir, dict_to_attrdict(cfg.patch_manager), dict_to_attrdict(cfg.val_data_keys), max(cfg.s_subgraph))
    dloader = iter(DataLoader(dset))
    acc_reward = 0
    forwarder = Forwarder()
    delta_dist = 0.4

    # segm_metric = AveragePrecision()
    clst_metric_rl = ClusterMetrics()
    # clst_metric = ClusterMetrics()
    metric_sp_gt = ClusterMetrics()
    # clst_metric_mcaff = ClusterMetrics()
    # clst_metric_mcembed = ClusterMetrics()
    # clst_metric_graphagglo = ClusterMetrics()
    sbd = SBD()

    # map_rl, map_embed, map_sp_gt, map_mcaff, map_mcembed, map_graphagglo = [], [], [], [], [], []
    sbd_rl, sbd_embed, sbd_sp_gt, sbd_mcaff, sbd_mcembed, sbd_graphagglo = [], [], [], [], [], []

    n_examples = len(dset)
    for it in range(n_examples):
        update_env_data(env, dloader, dset, device, with_gt_edges=False)
        env.reset()
        state = env.get_state()
        distr, _, _, _, _, node_features, embeddings = forwarder.forward(model, state, State,
                                                                              device,
                                                                              grad=False, post_data=False,
                                                                              get_node_feats=True,
                                                                              get_embeddings=True)
        action = torch.sigmoid(distr.loc)
        reward = env.execute_action(action, tau=0.0, train=False)
        acc_reward += reward[-2].item()

        embeds = embeddings[0].cpu()
        # node_features = node_features.cpu().numpy()
        rag = env.rags[0]
        edge_ids = rag.uvIds()
        gt_seg = env.gt_seg[0].cpu().numpy()
        # l2_embeddings = get_angles(embeds[None])[0]
        # l2_node_feats = get_angles(torch.from_numpy(node_features.T[None, ..., None])).squeeze().T.numpy()
        # clst_labels_kmeans = cluster_embeddings(l2_embeddings.permute((1, 2, 0)), len(np.unique(gt_seg)))
        # node_labels = cluster_embeddings(l2_node_feats, len(np.unique(gt_seg)))
        # clst_labels_sp_kmeans = elf.segmentation.features.project_node_labels_to_pixels(rag, node_labels).squeeze()

        # clst_labels_sp_graph_agglo = get_soln_graph_clustering(env.init_sp_seg, torch.from_numpy(edge_ids.astype(np.int)), torch.from_numpy(l2_node_feats), len(np.unique(gt_seg)))[0][0].numpy()
        # mc_labels_aff = env.get_current_soln(edge_weights=env.edge_features[:, 0]).cpu().numpy()[0]
        # ew_embedaffs = 1 - get_edge_features_1d(env.init_sp_seg[0].cpu().numpy(), offs, get_affinities_from_embeddings_2d(embeddings, offs, delta_dist, distance)[0].cpu().numpy())[0][:, 0]
        # mc_labels_embedding_aff = env.get_current_soln(edge_weights=torch.from_numpy(ew_embedaffs).to(device)).cpu().numpy()[0]
        rl_labels = env.current_soln.cpu().numpy()[0]

        ex_embeds.append(pca_project(embeds, n_comps=3))
        ex_raws.append(env.raw[0].cpu().permute(1, 2, 0).squeeze())
        # ex_sps.append(cm.prism(env.init_sp_seg[0].cpu() / env.init_sp_seg[0].max().item()))
        ex_sps.append(env.init_sp_seg[0].cpu())
        ex_mc_gts.append(project_overseg_to_seg(env.init_sp_seg[0], torch.from_numpy(gt_seg).to(device)).cpu().numpy())

        ex_gts.append(gt_seg)
        ex_rl.append(rl_labels)
        # ex_clst.append(clst_labels_kmeans)
        # ex_clst_sp.append(clst_labels_sp_kmeans)
        # ex_clst_graph_agglo.append(clst_labels_sp_graph_agglo)
        # ex_mcaff.append(mc_labels_aff)
        # ex_mc_embed.append(mc_labels_embedding_aff)

        # map_rl.append(segm_metric(rl_labels, gt_seg))
        sbd_rl.append(sbd(gt_seg, rl_labels))
        clst_metric_rl(rl_labels, gt_seg)

        # map_sp_gt.append(segm_metric(ex_mc_gts[-1], gt_seg))
        sbd_sp_gt.append(sbd(gt_seg, ex_mc_gts[-1]))
        metric_sp_gt(ex_mc_gts[-1], gt_seg)

        # map_embed.append(segm_metric(clst_labels_kmeans, gt_seg))
        # clst_metric(clst_labels_kmeans, gt_seg)

        # map_mcaff.append(segm_metric(mc_labels_aff, gt_seg))
        # sbd_mcaff.append(sbd(gt_seg, mc_labels_aff))
        # clst_metric_mcaff(mc_labels_aff, gt_seg)
        #
        # map_mcembed.append(segm_metric(mc_labels_embedding_aff, gt_seg))
        # sbd_mcembed.append(sbd(gt_seg, mc_labels_embedding_aff))
        # clst_metric_mcembed(mc_labels_embedding_aff, gt_seg)
        #
        # map_graphagglo.append(segm_metric(clst_labels_sp_graph_agglo, gt_seg))
        # sbd_graphagglo.append(sbd(gt_seg, clst_labels_sp_graph_agglo.astype(np.int)))
        # clst_metric_graphagglo(clst_labels_sp_graph_agglo.astype(np.int), gt_seg)

    print("\nSBD: ")
    print(f"sp gt       : {round(np.array(sbd_sp_gt).mean(), 4)}; {round(np.array(sbd_sp_gt).std(), 4)}")
    print(f"ours        : {round(np.array(sbd_rl).mean(), 4)}; {round(np.array(sbd_rl).std(), 4)}")
    # print(f"mc node     : {np.array(sbd_mcembed).mean()}")
    # print(f"mc embed    : {np.array(sbd_mcaff).mean()}")
    # print(f"graph agglo : {np.array(sbd_graphagglo).mean()}")

    # print("\nmAP: ")
    # print(f"sp gt       : {np.array(map_sp_gt).mean()}")
    # print(f"ours        : {np.array(map_rl).mean()}")
    # print(f"mc node     : {np.array(map_mcembed).mean()}")
    # print(f"mc embed    : {np.array(map_mcaff).mean()}")
    # print(f"graph agglo : {np.array(map_graphagglo).mean()}")
    #
    vi_rl_s, vi_rl_m, are_rl, arp_rl, arr_rl = clst_metric_rl.dump()
    vi_spgt_s, vi_spgt_m, are_spgt, arp_spgt, arr_spgt = metric_sp_gt.dump()
    # vi_mcaff_s, vi_mcaff_m, are_mcaff, arp_mcaff, arr_mcaff = clst_metric_mcaff.dump()
    # vi_mcembed_s, vi_mcembed_m, are_mcembed, arp_embed, arr_mcembed = clst_metric_mcembed.dump()
    # vi_graphagglo_s, vi_graphagglo_m, are_graphagglo, arp_graphagglo, arr_graphagglo = clst_metric_graphagglo.dump()
    #
    vi_rl_s_std, vi_rl_m_std, are_rl_std, arp_rl_std, arr_rl_std = clst_metric_rl.dump_std()
    vi_spgt_s_std, vi_spgt_m_std, are_spgt_std, arp_spgt_std, arr_spgt_std = metric_sp_gt.dump_std()

    print("\nVI merge: ")
    print(f"sp gt       : {round(vi_spgt_m, 4)}; {round(vi_spgt_m_std, 4)}")
    print(f"ours        : {round(vi_rl_m, 4)}; {round(vi_rl_m_std, 4)}")
    # print(f"mc affnties : {vi_mcaff_m}")
    # print(f"mc embed    : {vi_mcembed_m}")
    # print(f"graph agglo : {vi_graphagglo_m}")
    #
    print("\nVI split: ")
    print(f"sp gt       : {round(vi_spgt_s, 4)}; {round(vi_spgt_s_std, 4)}")
    print(f"ours        : {round(vi_rl_s, 4)}; {round(vi_rl_s_std, 4)}")
    # print(f"mc affnties : {vi_mcaff_s}")
    # print(f"mc embed    : {vi_mcembed_s}")
    # print(f"graph agglo : {vi_graphagglo_s}")
    #
    print("\nARE: ")
    print(f"sp gt       : {round(are_spgt, 4)}; {round(are_spgt_std, 4)}")
    print(f"ours        : {round(are_rl, 4)}; {round(are_rl_std, 4)}")
    # print(f"mc affnties : {are_mcaff}")
    # print(f"mc embed    : {are_mcembed}")
    # print(f"graph agglo : {are_graphagglo}")
    #
    print("\nARP: ")
    print(f"sp gt       : {round(arp_spgt, 4)}; {round(arp_spgt_std, 4)}")
    print(f"ours        : {round(arp_rl, 4)}; {round(arp_rl_std, 4)}")
    # print(f"mc affnties : {arp_mcaff}")
    # print(f"mc embed    : {arp_embed}")
    # print(f"graph agglo : {arp_graphagglo}")
    #
    print("\nARR: ")
    print(f"sp gt       : {round(arr_spgt, 4)}; {round(arr_spgt_std, 4)}")
    print(f"ours        : {round(arr_rl, 4)}; {round(arr_rl_std, 4)}")
    # print(f"mc affnties : {arr_mcaff}")
    # print(f"mc embed    : {arr_mcembed}")
    # print(f"graph agglo : {arr_graphagglo}")

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
    baselinefile = z5py.File('/scratch/pape/embeddings/leptin/data_16.n5')['default_with_boundaries']
    baselinefile[f'im000']['embed_sp']
    wandb.init(project="dbg", entity="aule", config="./default_configs.yaml")
    cfg = wandb.config
    device = torch.device("cuda:0")
    distance = CosineDistance()
    # fe_ext = FeExtractor(dict_to_attrdict(cfg.backbone), distance, device)
    # fe_ext.embed_model.load_state_dict(torch.load(cfg.fe_model_name))
    # fe_ext.cuda(device)
    env = MulticutEmbeddingsEnv(cfg, device)
    model = Agent(cfg, State, distance, device)
    model.load_state_dict(torch.load(cfg.agent_model_name))
    model.cuda(device)

    validate_and_compare_to_clustering(model, env, distance, device, cfg)