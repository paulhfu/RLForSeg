from data.spg_dset import SpgDset
import torch
import elf
import omegaconf
from torch.nn import functional as F
from torch.utils.data import DataLoader
from collections import namedtuple
from environments.multicut import MulticutEmbeddingsEnv
from environments.embedding_space_edge import EmbeddingSpaceEnvEdgeBased
from environments.embedding_space_node import EmbeddingSpaceEnvNodeBased
from models.agent_model import Agent
from models.feature_extractor import FeExtractor
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tensorboardX import SummaryWriter
import os
import numpy as np
from utils.exploration_functions import RunningAverage
from utils.general import get_angles, adjust_learning_rate, soft_update_params, set_seed_everywhere, cluster_embeddings, pca_project
from utils.matching import matching
from utils.replay_memory import TransitionData_ts
from utils.graphs import get_joint_sg_logprobs_edges, get_joint_sg_logprobs_nodes
from utils.distances import CosineDistance
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yaml
import portalocker
from shutil import copyfile
from torch.optim.lr_scheduler import ReduceLROnPlateau
from losses.contrastive_loss import ContrastiveLoss
from losses.rag_contrastive_loss import RagContrastiveLoss
from losses.rag_infonce_loss import RagInfoNceLoss
from losses.affinity_contrastive_loss import AffinityContrastive


class AgentSacTrainer(object):

    def __init__(self, cfg, global_count, global_writer_loss_count,
                 global_writer_quality_count, action_stats_count, global_writer_count, save_dir):
        super(AgentSacTrainer, self).__init__()

        self.cfg = cfg
        self.global_count = global_count
        self.global_writer_loss_count = global_writer_loss_count
        self.global_writer_quality_count = global_writer_quality_count
        self.action_stats_count = action_stats_count
        self.global_writer_count = global_writer_count
        self.memory = TransitionData_ts(capacity=self.cfg.trainer.t_max)
        self.save_dir = save_dir
        if 'nodes' in cfg.gen.env:
            self.get_joint_sg_logprobs = get_joint_sg_logprobs_nodes
        else:
            self.get_joint_sg_logprobs = get_joint_sg_logprobs_edges


    def setup(self, rank, world_size):
        # BLAS setup
        os.environ['OMP_NUM_THREADS'] = '10'
        os.environ['MKL_NUM_THREADS'] = '10'

        # assert torch.cuda.device_count() == 1
        torch.set_default_tensor_type('torch.FloatTensor')

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = self.cfg.gen.master_port

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    def validate(self, model, env, device):
        """validates the prediction against the method of clustering the embedding space"""

        model.eval()
        n_examples = 4
        taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        dset = SpgDset(self.cfg.gen.data_dir, max(self.cfg.sac.s_subgraph), self.cfg.gen.patch_manager, self.cfg.gen.patch_stride, self.cfg.gen.patch_shape, self.cfg.gen.reorder_sp)
        dloader = DataLoader(dset, batch_size=1, shuffle=True, pin_memory=True,
                             num_workers=0)
        clst_scores, clst_scores_sp, rl_scores, keys = [], [], [], None
        ex_raws, ex_sps, ex_gts, ex_embeds, ex_clst, ex_clst_sp, ex_rl = [], [], [], [], [], [], []
        for it in range(self.cfg.gen.validation.n_data_points):
            self.update_env_data(env, dloader, device)
            env.reset()
            state = env.get_state()
            _, _, _, action, embeddings, _, node_features = self.agent_forward(env, model, state=state, grad=False,
                                                           post_input=False, post_model=False, return_node_features=True)
            env.execute_action(action, None, post_stats=False)

            clst_labels = cluster_embeddings(embeddings.cpu().numpy().squeeze().transpose((1, 2, 0)), len(torch.unique(env.gt_seg)))
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
                    clst_scores_sp[tau_it] += np.array(list(map(float, list(_clst_scores_sp[tau_it]._asdict().values())[1:])))
                    rl_scores[tau_it] += np.array(list(map(float, list(_rl_scores[tau_it]._asdict().values())[1:])))

        div = np.ones_like(clst_scores[0])
        for i, key in enumerate(keys):
            if key not in ('fp', 'tp', 'fn'):
                div[i] = self.cfg.gen.validation.n_data_points

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

    def pretrain_embeddings(self, model, device, writer=None):
        wu_cfg = self.cfg.fe.warmup
        dset = SpgDset(self.cfg.gen.data_dir, max(self.cfg.sac.s_subgraph), wu_cfg.patch_manager, wu_cfg.patch_stride, wu_cfg.patch_shape, wu_cfg.reorder_sp)
        dloader = DataLoader(dset, batch_size=wu_cfg.batch_size, shuffle=True, pin_memory=True, num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=wu_cfg.lr,  betas=wu_cfg.betas)
        sheduler = ReduceLROnPlateau(optimizer)
        if wu_cfg.method == 'superpixel_contrast':
            criterion = RagContrastiveLoss(delta_var=self.cfg.fe.contrastive_delta_var,
                                           delta_dist=self.cfg.fe.contrastive_delta_dist,
                                           distance=self.distance)
        elif wu_cfg.method == 'affinity_contrast':
            criterion = AffinityContrastive(delta_var=0.1, delta_dist=0.3)
        acc_loss = 0
        iteration = 0

        while iteration <= wu_cfg.n_iterations:
            for it, (raw, gt, sp_seg, indices) in enumerate(dloader):
                raw, gt, sp_seg = raw.to(device), gt.to(device), sp_seg.to(device)
                edges = dloader.dataset.get_graphs(indices, sp_seg, device)[0]

                off = 0
                for i in range(len(edges)):
                    sp_seg[i] += off
                    edges[i] += off
                    off = int(sp_seg[i].max()) + 1
                edges = torch.cat(edges, 1)
                embeddings = model(raw).unsqueeze(2)
                # put embeddings on unit sphere so we can use cosine distance
                embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)

                loss = criterion(embeddings=embeddings, gt=gt, sp_seg=sp_seg.unsqueeze(2), edges=edges, raw=raw)

                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                optimizer.step()
                acc_loss += loss.item()

                if writer is not None:
                    writer.add_scalar("fe_warm_start/loss", loss.item(), iteration)
                    writer.add_scalar("fe_warm_start/lr", optimizer.param_groups[0]['lr'], iteration)
                if it % 10 == 0:
                    sheduler.step(acc_loss / 10)
                    acc_loss = 0
                iteration += 1
                if iteration % 100 == 0:
                    model.post_pca(get_angles(embeddings.squeeze(2).detach())[0].cpu(), tag="image/pix_embedding_proj", writer=False)
                if iteration > self.cfg.fe.warmup.n_iterations:
                    break

                del loss
                del embeddings
        return

    def cleanup(self):
        dist.destroy_process_group()

    def update_env_data(self, env, dloader, device):
        raw, gt, sp_seg, indices = next(iter(dloader))
        raw, gt, sp_seg = raw.to(device), gt.to(device), sp_seg.to(device)
        ret = dloader.dataset.get_graphs(indices, sp_seg, device)
        while ret == None:
            raw, gt, sp_seg, indices = next(iter(dloader))
            raw, gt, sp_seg = raw.to(device), gt.to(device), sp_seg.to(device)
            ret = dloader.dataset.get_graphs(indices, sp_seg, device)
        edges, edge_feat, _, gt_edges  = ret
        env.update_data(edge_ids=edges, gt_edges=gt_edges, edge_features=edge_feat, sp_seg=sp_seg, raw=raw, gt=gt)

    def agent_forward(self, env, model, state, actions=None, grad=True, post_input=False, post_model=False,
                      policy_opt=False, embeddings_opt=False, return_node_features=False):
        with torch.set_grad_enabled(grad):
            state = self.state_to_cuda(state, env.device, env.State)
            if actions is not None:
                actions = actions.to(model.module.device)
            ret = model(state,
                        actions,
                        post_input,
                        policy_opt and grad,
                        embeddings_opt,
                        return_node_features)

            if post_model and grad:
                for name, value in model.module.actor.named_parameters():
                    model.writer.add_histogram(name, value.data.cpu().numpy(), self.global_count.value())
                    model.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.global_count.value())
                for name, value in model.module.critic_tgt.named_parameters():
                    model.writer.add_histogram(name, value.data.cpu().numpy(), self.global_count.value())
                    model.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.global_count.value())

        return ret

    def update_critic(self, obs, action, reward, next_obs, not_done, env, model, optimizers):
        distribution, target_Q1, target_Q2, next_action, _, side_loss = self.agent_forward(env, model, state=next_obs)
        current_Q1, current_Q2, side_loss = self.agent_forward(env, model, state=obs, actions=action)

        log_prob = distribution.log_prob(next_action)
        critic_loss = torch.tensor([0.0], device=target_Q1[0].device)
        mean_reward = 0

        for i, sz in enumerate(self.cfg.sac.s_subgraph):
            _log_prob, sg_entropy = self.get_joint_sg_logprobs(log_prob, distribution.scale, next_obs, i , sz)
            if self.cfg.sac.use_closed_form_entropy:
                target_V = torch.min(target_Q1[i], target_Q2[i]) - model.module.alpha[i].detach() * sg_entropy
            else:
                target_V = torch.min(target_Q1[i], target_Q2[i]) - model.module.alpha[i].detach() * _log_prob

            target_Q = reward[i] + (not_done * self.cfg.sac.discount * target_V)
            target_Q = target_Q.detach()

            critic_loss = critic_loss + (F.mse_loss(current_Q1[i], target_Q) + F.mse_loss(current_Q2[i], target_Q))# / 2) + self.cfg.sac.sl_beta * side_loss
            mean_reward += reward[i].mean()
        critic_loss = critic_loss / len(self.cfg.sac.s_subgraph)
        optimizers.critic.zero_grad()
        critic_loss.backward()
        optimizers.critic.step()

        return critic_loss.item(), mean_reward / len(self.cfg.sac.s_subgraph)

    def update_actor_and_alpha(self, obs, reward, env, model, optimizers, embeddings_opt=False):
        distribution, actor_Q1, actor_Q2, action, side_loss = \
            self.agent_forward(env, model, state=obs, policy_opt=True, embeddings_opt=embeddings_opt)

        log_prob = distribution.log_prob(action)
        actor_loss = torch.tensor([0.0], device=actor_Q1[0].device)
        alpha_loss = torch.tensor([0.0], device=actor_Q1[0].device)
        _log_prob, sg_entropy = [], []
        for i, sz in enumerate(self.cfg.sac.s_subgraph):
            actor_Q = torch.min(actor_Q1[i], actor_Q2[i])
            ret = self.get_joint_sg_logprobs(log_prob, distribution.scale, obs, i , sz)
            _log_prob.append(ret[0])
            sg_entropy.append(ret[1])

            loss = (model.module.alpha[i].detach() * _log_prob[i] - actor_Q).mean()
            actor_loss = actor_loss + loss# + self.cfg.sac.sl_beta * side_loss

        actor_loss = actor_loss / len(self.cfg.sac.s_subgraph)
        optimizers.actor.zero_grad()
        actor_loss.backward()
        optimizers.actor.step()

        min_entropy = (self.cfg.sac.entropy_range[1] - self.cfg.sac.entropy_range[0]) * ((1.5-reward[-1]) / 1.5) \
                      + self.cfg.sac.entropy_range[0]

        for i, sz in enumerate(self.cfg.sac.s_subgraph):
            min_entropy = min_entropy.to(model.module.alpha[i].device).squeeze()
            if self.cfg.sac.use_closed_form_entropy:
                alpha_loss = alpha_loss + (model.module.alpha[i] \
                                           * (sg_entropy[i].detach() - (self.cfg.sac.s_subgraph[i] * min_entropy))).mean()
            else:
                alpha_loss = alpha_loss + (model.module.alpha[i] *
                                           (-_log_prob[i].detach() - (self.cfg.sac.s_subgraph[i] * min_entropy))).mean()

        alpha_loss = alpha_loss / len(self.cfg.sac.s_subgraph)
        optimizers.temperature.zero_grad()
        alpha_loss.backward()
        optimizers.temperature.step()

        return actor_loss.item(), alpha_loss.item()

    def _step(self, replay_buffer, optimizers, mov_sum_loss, env, model, step, writer=None):

        (obs, action, reward, next_obs, done), sample_idx = replay_buffer.sample()
        not_done = int(not done)
        embeddings_opt = step - self.cfg.fe.n_prep_steps > 0 and (step - self.cfg.fe.n_prep_steps) % self.cfg.fe.update_frequency == 0

        critic_loss, mean_reward = self.update_critic(obs, action, reward, next_obs, not_done, env, model, optimizers)
        replay_buffer.report_sample_loss(critic_loss + mean_reward, sample_idx)

        actor_loss, alpha_loss = self.update_actor_and_alpha(obs, reward, env, model, optimizers, embeddings_opt)

        mov_sum_loss.critic.apply(critic_loss)
        mov_sum_loss.actor.apply(actor_loss)
        mov_sum_loss.temperature.apply(alpha_loss)

        if self.global_writer_loss_count.value() > self.cfg.trainer.lr_sched.mov_avg_bandwidth \
                and self.global_writer_loss_count.value() % self.cfg.trainer.lr_sched.step_frequency == 0:
            optimizers.critic_shed.step(mov_sum_loss.critic.avg)
            optimizers.actor_shed.step(mov_sum_loss.actor.avg)
            optimizers.temp_shed.step(mov_sum_loss.actor.avg)  # actor and temp should have the same lr for primal dual iteration
            writer.add_scalar("mov_sum/critic", mov_sum_loss.critic.avg, self.global_writer_loss_count.value())
            writer.add_scalar("mov_sum/actor", mov_sum_loss.actor.avg, self.global_writer_loss_count.value())
            writer.add_scalar("mov_sum/temperature", mov_sum_loss.temperature.avg, self.global_writer_loss_count.value())

        writer.add_scalar("loss/critic", critic_loss, self.global_writer_loss_count.value())
        writer.add_scalar("loss/actor", actor_loss, self.global_writer_loss_count.value())
        writer.add_scalar("loss/temperature", alpha_loss, self.global_writer_loss_count.value())
        writer.add_scalar("lr/critic", optimizers.critic_shed.optimizer.param_groups[0]['lr'], self.global_writer_loss_count.value())
        writer.add_scalar("lr/actor", optimizers.actor_shed.optimizer.param_groups[0]['lr'], self.global_writer_loss_count.value())
        writer.add_scalar("lr/temperature", optimizers.temp_shed.optimizer.param_groups[0]['lr'], self.global_writer_loss_count.value())


        if step % self.cfg.sac.critic_target_update_frequency == 0:
            soft_update_params(model.module.critic, model.module.critic_tgt, self.cfg.sac.critic_tau)


        self.global_writer_loss_count.increment()

    # Acts and trains model
    def train(self, rank, return_dict, rn):

        self.log_dir = os.path.join(self.save_dir, 'logs', '_' + str(rn))
        writer = None
        if rank == 0:
            writer = SummaryWriter(logdir=self.log_dir)
            writer.add_text("conf", omegaconf.OmegaConf.to_yaml(self.cfg))
            # writer.add_text("conf", self.cfg.pretty())
            copyfile(os.path.join(self.save_dir, 'runtime_cfg.yaml'),
                     os.path.join(self.log_dir, 'runtime_cfg.yaml'))

            self.global_count.reset()
            self.global_writer_loss_count.reset()
            self.global_writer_quality_count.reset()
            self.action_stats_count.reset()
            self.global_writer_count.reset()

        set_seed_everywhere(rn)
        if rank == 0:
            print('training with seed: ' + str(rn))
        score = self.train_step(rank, writer)
        if rank == 0:
            return_dict['score'] = score
            del self.memory
        return

    def train_step(self, rank, writer):
        is_edge_based = True
        device = torch.device("cuda:" + str(rank // self.cfg.gen.n_processes_per_gpu))
        print('Running on device: ', device)
        torch.cuda.set_device(device)
        torch.set_default_tensor_type(torch.FloatTensor)
        self.setup(rank, self.cfg.gen.n_processes_per_gpu * self.cfg.gen.n_gpu)
        # cosine distance (input should always be normalized)
        self.distance = CosineDistance()

        fe_ext = FeExtractor(self.cfg.fe.n_raw_channels, self.cfg.fe.n_embedding_features,
                             self.distance, device, writer, self.cfg.fe.max_pixel_in_dist_mat)
        fe_ext.cuda(device)

        if self.cfg.gen.env == "multicut_embedding":
            env = MulticutEmbeddingsEnv(fe_ext, self.cfg, device, writer=writer, writer_counter=self.global_writer_quality_count)
        elif self.cfg.gen.env == "embedding_space_edges":
            env = EmbeddingSpaceEnvEdgeBased(fe_ext, self.cfg, device, writer=writer, writer_counter=self.global_writer_quality_count)
        elif self.cfg.gen.env == "embedding_space_nodes":
            is_edge_based = False
            env = EmbeddingSpaceEnvNodeBased(fe_ext, self.cfg, device, writer=writer, writer_counter=self.global_writer_quality_count)

        model = Agent(self.cfg, env.State, self.distance, device, writer=writer)
        model.cuda(device)
        #Create shared network
        shared_model = DDP(model, device_ids=[device], find_unused_parameters=True)
        if 'extra' in self.cfg.fe.optim:
            # optimizers
            MovSumLosses = namedtuple('mov_avg_losses', ('actor', 'embeddings', 'critic', 'temperature'))
            OptimizerContainer = namedtuple('OptimizerContainer', ('actor', 'embeddings', 'critic', 'temperature',
                                                                   'actor_shed', 'embed_shed', 'critic_shed',
                                                                   'temp_shed'))
        else:
            MovSumLosses = namedtuple('mov_avg_losses', ('actor', 'critic', 'temperature'))
            OptimizerContainer = namedtuple('OptimizerContainer', ('actor', 'critic', 'temperature',
                                                                   'actor_shed', 'critic_shed', 'temp_shed'))
        if "rl_loss" == self.cfg.fe.optim:
            actor_optimizer = torch.optim.Adam(list(shared_model.module.actor.parameters())
                                               + list(fe_ext.parameters()),
                                               lr=self.cfg.sac.actor_lr,
                                               betas=self.cfg.sac.actor_betas)
        else:
            actor_optimizer = torch.optim.Adam(shared_model.module.actor.parameters(),
                                               lr=self.cfg.sac.actor_lr,
                                               betas=self.cfg.sac.actor_betas)
        if "extra" in self.cfg.fe.optim:
            embeddings_optimizer = torch.optim.Adam(fe_ext.parameters(),
                                                    lr=self.cfg.fe.lr,
                                                    betas=self.cfg.fe.betas)
        critic_optimizer = torch.optim.Adam(shared_model.module.critic.parameters(),
                                            lr=self.cfg.sac.critic_lr,
                                            betas=self.cfg.sac.critic_betas)
        temp_optimizer = torch.optim.Adam([shared_model.module.log_alpha],
                                          lr=self.cfg.sac.alpha_lr,
                                          betas=self.cfg.sac.alpha_betas)

        bw = self.cfg.trainer.lr_sched.mov_avg_bandwidth
        off = self.cfg.trainer.lr_sched.mov_avg_offset
        weights = np.linspace(self.cfg.trainer.lr_sched.weight_range[0], self.cfg.trainer.lr_sched.weight_range[1], bw)
        weights = weights / weights.sum()  # make them sum up to one
        shed = self.cfg.trainer.lr_sched.torch_sched
        if "extra" in self.cfg.fe.optim:
            mov_sum_losses = MovSumLosses(RunningAverage(weights, band_width=bw, offset=off),
                                          RunningAverage(weights, band_width=bw, offset=off),
                                          RunningAverage(weights, band_width=bw, offset=off),
                                          RunningAverage(weights, band_width=bw, offset=off))
            optimizers = OptimizerContainer(actor_optimizer, embeddings_optimizer, critic_optimizer, temp_optimizer,
                                            *[ReduceLROnPlateau(opt, patience=shed.patience,
                                                                threshold=shed.threshold, min_lr=shed.min_lr,
                                                                factor=shed.factor) for opt in (
                                                  actor_optimizer, embeddings_optimizer, critic_optimizer,
                                                  temp_optimizer)])
        else:
            mov_sum_losses = MovSumLosses(RunningAverage(weights, band_width=bw, offset=off),
                                          RunningAverage(weights, band_width=bw, offset=off),
                                          RunningAverage(weights, band_width=bw, offset=off))
            optimizers = OptimizerContainer(actor_optimizer, critic_optimizer, temp_optimizer,
                                            *[ReduceLROnPlateau(opt, patience=shed.patience,
                                                                threshold=shed.threshold, min_lr=shed.min_lr,
                                                                factor=shed.factor) for opt in
                                              (actor_optimizer, critic_optimizer, temp_optimizer)])

        dist.barrier()

        if self.cfg.gen.resume:
            shared_model.module.load_state_dict(torch.load(os.path.join(self.save_dir, self.cfg.gen.model_name)))
            if self.cfg.gen.validation is not None:
                self.validate(model, env, device)
                return
        elif self.cfg.fe.load_pretrained:
            fe_ext.load_state_dict(torch.load(os.path.join(self.save_dir, self.cfg.fe.model_name)))
        elif 'warmup' in self.cfg.fe and rank == 0:
            print('pretrain fe extractor')
            if self.cfg.fe.warmup.method == "unsupervised":
                assert False  # not working yet
                self.pretrain_embeddings_prot_nce(fe_ext, device, writer)
            else:
                self.pretrain_embeddings(fe_ext, device, writer)
            torch.save(fe_ext.state_dict(),
                       os.path.join(self.save_dir, self.cfg.fe.model_name))
        dist.barrier()
        if "none" == self.cfg.fe.optim:
            for param in fe_ext.parameters():
                param.requires_grad = False

        dset = SpgDset(self.cfg.gen.data_dir, max(self.cfg.sac.s_subgraph), self.cfg.gen.patch_manager, self.cfg.gen.patch_stride, self.cfg.gen.patch_shape, self.cfg.gen.reorder_sp)
        step = 0

        while self.global_count.value() <= self.cfg.trainer.T_max:
            dloader = DataLoader(dset, batch_size=self.cfg.trainer.batch_size, shuffle=True, pin_memory=True,
                                 num_workers=0)
            for iteration in range(len(dset) * self.cfg.trainer.data_update_frequency):
                if iteration % self.cfg.trainer.data_update_frequency == 0:
                    self.update_env_data(env, dloader, device)
                env.reset()
                self.update_rt_vars(critic_optimizer, actor_optimizer)
                if rank == 0 and self.cfg.rt_vars.safe_model:
                    if self.cfg.gen.model_name != "":
                        torch.save(shared_model.module.state_dict(),
                                   os.path.join(self.log_dir, self.cfg.gen.model_name))
                    else:
                        torch.save(shared_model.module.state_dict(), os.path.join(self.log_dir, 'agent_model'))

                state = env.get_state()
                while not env.done:
                    # Calculate policy and values
                    post_stats = True if (self.global_writer_count.value()) % self.cfg.trainer.post_stats_frequency == 0 \
                        else False
                    post_model = True if (self.global_writer_count.value() + 1) % self.cfg.trainer.post_model_frequency == 0 \
                        else False
                    post_stats &= self.memory.is_full()
                    post_model &= self.memory.is_full()
                    distr = None
                    if not self.memory.is_full():
                        if is_edge_based:
                            action = torch.rand((env.edge_ids.shape[-1], self.cfg.sac.n_actions), device=device)
                        else:
                            action = torch.rand((env.current_node_embeddings.shape[0], self.cfg.sac.n_actions), device=device)
                        action *= self.cfg.sac.diag_gaussian_actor.sample_factor
                        action += self.cfg.sac.diag_gaussian_actor.sample_offset
                    else:
                        distr, _, _, action, _, _ = self.agent_forward(env, shared_model, state=state, grad=False,
                                                                       post_input=post_stats, post_model=post_model)

                    logg_dict = {}
                    if post_stats:
                        for i in range(len(self.cfg.sac.s_subgraph)):
                            logg_dict['alpha_' + str(i)] = shared_model.module.alpha[i].item()
                        if distr is not None:
                            logg_dict['mean_loc'] = distr.loc.mean().item()
                            logg_dict['mean_scale'] = distr.scale.mean().item()

                    if self.memory.is_full():
                        for i in range(self.cfg.trainer.n_updates_per_step):
                            self._step(self.memory, optimizers, mov_sum_losses, env, shared_model,
                                       step, writer=writer)

                    next_state, reward = env.execute_action(action, logg_dict, post_stats=post_stats)

                    self.memory.push(self.state_to_cpu(state, env.State), action, reward, self.state_to_cpu(next_state, env.State), env.done)
                    state = next_state

                self.global_count.increment()
                step += 1
                if rank == 0:
                    self.global_writer_count.increment()
                if step > self.cfg.trainer.T_max:
                    break

        dist.barrier()
        if rank == 0:
            self.memory.clear()
            if self.cfg.gen.model_name != "":
                torch.save(shared_model.state_dict(), os.path.join(self.log_dir, self.cfg.gen.model_name))
            else:
                torch.save(shared_model.state_dict(), os.path.join(self.log_dir, 'agent_model'))
            print('saved')

        self.cleanup()
        return sum(env.acc_reward) / len(env.acc_reward)

    def state_to_cpu(self, state, state_class):
        state = list(state)
        for i in range(len(state)):
            if torch.is_tensor(state[i]):
                state[i] = state[i].cpu()
            elif isinstance(state[i], list) or isinstance(state[i], tuple):
                state[i] = self.state_to_cpu(state[i], None)
        if state_class is not None:
            return state_class(*state)
        return state

    def state_to_cuda(self, state, device, state_class):
        state = list(state)
        for i in range(len(state)):
            if torch.is_tensor(state[i]):
                state[i] = state[i].to(device)
            elif isinstance(state[i], list) or isinstance(state[i], tuple):
                state[i] = self.state_to_cuda(state[i], device, None)
        if state_class is not None:
            return state_class(*state)
        return state

    def update_rt_vars(self, critic_optimizer, actor_optimizer):
        with portalocker.Lock(os.path.join(self.log_dir, 'runtime_cfg.yaml'), 'rb+', timeout=60) as fh:
            with open(os.path.join(self.log_dir, 'runtime_cfg.yaml')) as info:
                args_dict = yaml.full_load(info)
                if args_dict is not None:
                    if 'safe_model' in args_dict:
                        self.cfg.rt_vars.safe_model = args_dict['safe_model']
                        args_dict['safe_model'] = False
                    if 'add_noise' in args_dict:
                        self.cfg.rt_vars.add_noise = args_dict['add_noise']
                    if 'critic_lr' in args_dict and args_dict['critic_lr'] != self.cfg.sac.critic_lr:
                        self.cfg.sac.critic_lr = args_dict['critic_lr']
                        adjust_learning_rate(critic_optimizer, self.cfg.sac.critic_lr)
                    if 'actor_lr' in args_dict and args_dict['actor_lr'] != self.cfg.sac.actor_lr:
                        self.cfg.sac.actor_lr = args_dict['actor_lr']
                        adjust_learning_rate(actor_optimizer, self.cfg.sac.actor_lr)
            with open(os.path.join(self.log_dir, 'runtime_cfg.yaml'), "w") as info:
                yaml.dump(args_dict, info)

            # flush and sync to filesystem
            fh.flush()
            os.fsync(fh.fileno())
