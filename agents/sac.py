import os
import torch
import elf
import numpy as np
import wandb
from elf.segmentation.features import compute_rag
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib import cm
from multiprocessing import Process, Lock
import threading
import shutil
from skimage.morphology import dilation

from environments.multicut import MulticutEmbeddingsEnv, State
from data.spg_dset import SpgDset
from models.agent_model import Agent
from utils.exploration_functions import RunningAverage
from utils.general import soft_update_params, set_seed_everywhere, get_colored_edges_in_sseg, pca_project, \
    random_label_cmap
from utils.replay_memory import TransitionData_ts
from utils.graphs import get_joint_sg_logprobs_edges
from utils.distances import CosineDistance, L2Distance
from utils.matching import matching
from utils.yaml_conv_parser import dict_to_attrdict
from utils.training_helpers import update_env_data, supervised_policy_pretraining, state_to_cpu, Forwarder
from utils.metrics import AveragePrecision, ClusterMetrics


# from timeit import default_timer as timer


class AgentSacTrainer(object):

    def __init__(self, cfg, global_count):
        super(AgentSacTrainer, self).__init__()
        assert torch.cuda.device_count() == 1
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        torch.set_default_tensor_type(torch.FloatTensor)

        self.cfg = cfg
        self.global_count = global_count
        self.memory = TransitionData_ts(capacity=self.cfg.mem_size)
        self.best_val_reward = -np.inf
        if self.cfg.distance == 'cosine':
            self.distance = CosineDistance()
        else:
            self.distance = L2Distance()

        self.model = Agent(self.cfg, State, self.distance, self.device)
        wandb.watch(self.model)
        self.model.cuda(self.device)
        self.model_mtx = Lock()

        MovSumLosses = namedtuple('mov_avg_losses', ('actor', 'critic', 'temperature'))
        Scalers = namedtuple('Scalers', ('critic', 'actor'))
        OptimizerContainer = namedtuple('OptimizerContainer',
                                        ('actor', 'critic', 'temperature', 'actor_shed', 'critic_shed', 'temp_shed'))
        actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=self.cfg.actor_lr)
        if self.cfg.fe_optimization:
            critic_optimizer = torch.optim.Adam(list(self.model.critic.parameters()) + list(self.model.fe_ext.parameters()), lr=self.cfg.critic_lr)
        else:
            critic_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=self.cfg.critic_lr)
        # critic_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=self.cfg.critic_lr)
        temp_optimizer = torch.optim.Adam([self.model.log_alpha], lr=self.cfg.alpha_lr)

        lr_sched_cfg = dict_to_attrdict(self.cfg.lr_sched)
        bw = lr_sched_cfg.mov_avg_bandwidth
        off = lr_sched_cfg.mov_avg_offset
        weights = np.linspace(lr_sched_cfg.weight_range[0], lr_sched_cfg.weight_range[1], bw)
        weights = weights / weights.sum()  # make them sum up to one
        shed = lr_sched_cfg.torch_sched

        self.mov_sum_losses = MovSumLosses(RunningAverage(weights, band_width=bw, offset=off),
                                           RunningAverage(weights, band_width=bw, offset=off),
                                           RunningAverage(weights, band_width=bw, offset=off))
        self.optimizers = OptimizerContainer(actor_optimizer, critic_optimizer, temp_optimizer,
                                             *[ReduceLROnPlateau(opt, patience=shed.patience,
                                                                 threshold=shed.threshold, min_lr=shed.min_lr,
                                                                 factor=shed.factor) for opt in
                                               (actor_optimizer, critic_optimizer, temp_optimizer)])
        self.scalers = Scalers(torch.cuda.amp.GradScaler(), torch.cuda.amp.GradScaler())
        self.forwarder = Forwarder()

        if self.cfg.agent_model_name != "":
            self.model.load_state_dict(torch.load(self.cfg.agent_model_name))
        # finished with prepping

        self.train_dset = SpgDset(self.cfg.data_dir, dict_to_attrdict(self.cfg.patch_manager),
                                  dict_to_attrdict(self.cfg.train_data_keys), max(self.cfg.s_subgraph))
        self.val_dset = SpgDset(self.cfg.val_data_dir, dict_to_attrdict(self.cfg.patch_manager),
                                dict_to_attrdict(self.cfg.val_data_keys), max(self.cfg.s_subgraph))

        self.segm_metric = AveragePrecision()
        self.clst_metric = ClusterMetrics()
        self.global_counter = 0

    def validate(self):
        """validates the prediction against the method of clustering the embedding space"""
        env = MulticutEmbeddingsEnv(self.cfg, self.device)
        if self.cfg.verbose:
            print("\n\n###### start validate ######", end='')
        self.model.eval()
        n_examples = len(self.val_dset)
        # taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # rl_scores, keys = [], None

        self.clst_metric.reset()
        map_scores = []
        ex_raws, ex_sps, ex_gts, ex_mc_gts, ex_feats, ex_emb, ex_n_emb, ex_rl, edge_ids, rewards, actions = [[] for _ in
                                                                                                             range(11)]
        dloader = iter(DataLoader(self.val_dset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0))
        acc_reward = 0

        for it in range(n_examples):
            update_env_data(env, dloader, self.val_dset, self.device,
                            with_gt_edges="sub_graph_dice" in self.cfg.reward_function)
            env.reset()
            state = env.get_state()

            self.model_mtx.acquire()
            try:
                distr, _, _, _, _, node_features, embeddings = self.forwarder.forward(self.model, state, State,
                                                                                      self.device,
                                                                                      grad=False, post_data=False,
                                                                                      get_node_feats=True,
                                                                                      get_embeddings=True)
            finally:
                self.model_mtx.release()
            action = torch.sigmoid(distr.loc)
            reward = env.execute_action(action, tau=0.0, train=False)
            rew = reward[-1].item() if self.cfg.reward_function == "sub_graph_dice" else reward[-2].item()
            acc_reward += rew
            rl_labels = env.current_soln.cpu().numpy()[0]
            gt_seg = env.gt_seg[0].cpu().numpy()
            if self.cfg.verbose:
                print(f"\nstep: {it}; mean_loc: {round(distr.loc.mean().item(), 5)}; mean reward: {round(rew, 5)}",
                      end='')

            if it in self.cfg.store_indices:
                node_features = node_features[:env.n_offs[1]][env.init_sp_seg[0].long()].permute(2, 0, 1).cpu()
                gt_mc = cm.prism(env.gt_soln[0].cpu() / env.gt_soln[
                    0].max().item()) if env.gt_edge_weights is not None else torch.zeros(env.raw.shape[-2:])

                ex_feats.append(pca_project(node_features, n_comps=3))
                ex_emb.append(pca_project(embeddings[0].cpu(), n_comps=3))
                ex_n_emb.append(pca_project(node_features[:self.cfg.dim_embeddings], n_comps=3))
                ex_raws.append(env.raw[0].cpu().permute(1, 2, 0).squeeze())
                ex_sps.append(env.init_sp_seg[0].cpu())
                ex_mc_gts.append(gt_mc)
                ex_gts.append(gt_seg)
                ex_rl.append(rl_labels)
                edge_ids.append(env.edge_ids)
                rewards.append(reward[-1])
                actions.append(action)

            map_scores.append(self.segm_metric(rl_labels, gt_seg))
            self.clst_metric(rl_labels, gt_seg)

            '''
            _rl_scores = matching(gt_seg, rl_labels, thresh=taus, criterion='iou', report_matches=False)
            if it == 0:
                for tau_it in range(len(_rl_scores)):
                    rl_scores.append(np.array(list(map(float, list(_rl_scores[tau_it]._asdict().values())[1:]))))
                keys = list(_rl_scores[0]._asdict().keys())[1:]
            else:
                for tau_it in range(len(_rl_scores)):
                    rl_scores[tau_it] += np.array(list(map(float, list(_rl_scores[tau_it]._asdict().values())[1:]))
            '''

        '''
        div = np.ones_like(rl_scores[0])
        for i, key in enumerate(keys):
            if key not in ('fp', 'tp', 'fn'):
                div[i] = 10
        for tau_it in range(len(rl_scores)):
            rl_scores[tau_it] = dict(zip(keys, rl_scores[tau_it] / div))
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        plt.subplots_adjust(hspace=.5)
        for m in ('precision', 'recall', 'accuracy', 'f1'):
            y = [s[m] for s in rl_scores]
            data = [[x, y] for (x, y) in zip(taus, y)]
            table = wandb.Table(data=data, columns=["IoU_threshold", m])
            wandb.log({"validation/" + m: wandb.plot.line(table, "IoU_threshold", m, stroke=None, title=m)})
            axs[0].plot(taus, [s[m] for s in rl_scores], '.-', lw=2, label=m)
        axs[0].set_ylabel('Metric value')
        axs[0].grid()
        axs[0].legend(bbox_to_anchor=(.8, 1.65), loc='upper left', fontsize='xx-small')
        axs[0].set_title('RL method')
        axs[0].set_xlabel(r'IoU threshold $\tau$')
        for m in ('fp', 'tp', 'fn'):
            y = [s[m] for s in rl_scores]
            data = [[x, y] for (x, y) in zip(taus, y)]
            table = wandb.Table(data=data, columns=["IoU_threshold", m])
            wandb.log({"validation/" + m: wandb.plot.line(table, "IoU_threshold", m, stroke=None, title=m)})
            axs[1].plot(taus, [s[m] for s in rl_scores], '.-', lw=2, label=m)
        axs[1].set_ylabel('Number #')
        axs[1].grid()
        axs[1].legend(bbox_to_anchor=(.87, 1.6), loc='upper left', fontsize='xx-small');
        axs[1].set_title('RL method')
        axs[1].set_xlabel(r'IoU threshold $\tau$')
        #wandb.log({"validation/metrics": [wandb.Image(fig, caption="metrics")]})
        plt.close('all')
        '''

        splits, merges, are, arp, arr = self.clst_metric.dump()
        wandb.log({"validation/acc_reward": acc_reward})
        wandb.log({"validation/mAP": np.mean(map_scores)}, step=self.global_counter)
        wandb.log({"validation/UnderSegmVI": splits}, step=self.global_counter)
        wandb.log({"validation/OverSegmVI": merges}, step=self.global_counter)
        wandb.log({"validation/ARE": are}, step=self.global_counter)
        wandb.log({"validation/ARP": arp}, step=self.global_counter)
        wandb.log({"validation/ARR": arr}, step=self.global_counter)

        # do the lr sheduling
        self.optimizers.critic_shed.step(acc_reward)
        self.optimizers.actor_shed.step(acc_reward)

        if acc_reward > self.best_val_reward:
            self.best_val_reward = acc_reward
            wandb.run.summary["validation/acc_reward"] = acc_reward
            torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, "best_checkpoint_agent.pth"))
        if self.cfg.verbose:
            print("\n###### finish validate ######\n", end='')

        label_cm = random_label_cmap(zeroth=1.0)
        label_cm.set_bad(alpha=0)

        for it, i in enumerate(self.cfg.store_indices):
            fig, axs = plt.subplots(2, 4 if self.cfg.reward_function == "sub_graph_dice" else 5, sharex='col',
                                    figsize=(9, 5), sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
            axs[0, 0].imshow(ex_gts[it], cmap=random_label_cmap(), interpolation="none")
            axs[0, 0].set_title('gt', y=1.05, size=10)
            axs[0, 0].axis('off')
            if ex_raws[it].ndim == 3:
                if ex_raws[it].shape[-1] > 2:
                    axs[0, 1].imshow(ex_raws[it][..., :3], cmap="gray")
                else:
                    axs[0, 1].imshow(ex_raws[it][..., 0], cmap="gray")
            else:
                axs[1, 1].imshow(ex_raws[it], cmap="gray")

            axs[0, 1].set_title('raw image', y=1.05, size=10)
            axs[0, 1].axis('off')
            if ex_raws[it].ndim == 3:
                if ex_raws[it].shape[-1] > 1:
                    axs[0, 2].imshow(ex_raws[it][..., -1], cmap="gray")
                else:
                    axs[0, 2].imshow(ex_raws[it][..., 0], cmap="gray")
            else:
                axs[0, 2].imshow(ex_raws[it], cmap="gray")
            axs[0, 2].set_title('plantseg', y=1.05, size=10)
            axs[0, 2].axis('off')
            axs[0, 3].imshow(ex_sps[it], cmap=random_label_cmap(), interpolation="none")
            axs[0, 3].set_title('superpixels', y=1.05, size=10)
            axs[0, 3].axis('off')

            axs[1, 0].imshow(ex_feats[it])
            axs[1, 0].set_title('features', y=-0.15, size=10)
            axs[1, 0].axis('off')
            axs[1, 1].imshow(ex_n_emb[it])
            axs[1, 1].set_title('node embeddings', y=-0.15, size=10)
            axs[1, 1].axis('off')
            axs[1, 2].imshow(ex_emb[it])
            axs[1, 2].set_title('embeddings', y=-0.15, size=10)
            axs[1, 2].axis('off')
            axs[1, 3].imshow(ex_rl[it], cmap=random_label_cmap(), interpolation="none")
            axs[1, 3].set_title('prediction', y=-0.15, size=10)
            axs[1, 3].axis('off')

            if self.cfg.reward_function != "sub_graph_dice":
                frame_rew, scores_rew, bnd_mask = get_colored_edges_in_sseg(ex_sps[it][None].float(),
                                                                            edge_ids[it].cpu(), rewards[it].cpu())
                frame_act, scores_act, _ = get_colored_edges_in_sseg(ex_sps[it][None].float(), edge_ids[it].cpu(),
                                                                     1 - actions[it].cpu().squeeze())

                bnd_mask = torch.from_numpy(dilation(bnd_mask.cpu().numpy()))
                frame_rew = np.stack([dilation(frame_rew.cpu().numpy()[..., i]) for i in range(3)], -1)
                frame_act = np.stack([dilation(frame_act.cpu().numpy()[..., i]) for i in range(3)], -1)
                ex_rl[it] = ex_rl[it].squeeze().astype(np.float)
                ex_rl[it][bnd_mask] = np.nan

                axs[1, 4].imshow(frame_rew, interpolation="none")
                axs[1, 4].imshow(ex_rl[it], cmap=label_cm, alpha=0.8, interpolation="none")
                axs[1, 4].set_title("rewards", y=-0.2)
                axs[1, 4].axis('off')

                axs[0, 4].imshow(frame_act, interpolation="none")
                axs[0, 4].imshow(ex_rl[it], cmap=label_cm, alpha=0.8, interpolation="none")
                axs[0, 4].set_title("actions", y=1.05)
                axs[0, 4].axis('off')

            wandb.log({"validation/sample_" + str(i): [wandb.Image(fig, caption="sample images")]},
                      step=self.global_counter)
            plt.close('all')

    def update_critic(self, obs, next_obs, action, reward, not_done):
        self.optimizers.critic.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            distribution, target_Q, _, _, side_loss = self.forwarder.forward(self.model, next_obs, State, self.device, grad=False)
            critic_loss = torch.tensor([0.0], device=target_Q[0].device)
            mean_reward = 0
            log_prob = distribution.log_prob(action)
            current_Q, side_loss = self.forwarder.forward(self.model, obs, State, self.device, actions=action, grad=True)
            for i, sz in enumerate(self.cfg.s_subgraph):
                log_prob = get_joint_sg_logprobs_edges(log_prob, distribution.scale, obs, i, sz)[0]
                target_V = target_Q[i].detach() - self.model.alpha[i].detach() * log_prob
                tgt_Q = reward[i] + (not_done * 0.99 * target_V)

                critic_loss = critic_loss + F.mse_loss(current_Q[i], tgt_Q)
                mean_reward += reward[i].mean()
            critic_loss = critic_loss / len(self.cfg.s_subgraph) + self.cfg.side_loss_weight * side_loss

        self.scalers.critic.scale(critic_loss).backward()
        self.scalers.critic.step(self.optimizers.critic)
        self.scalers.critic.update()

        return critic_loss.item(), mean_reward / len(self.cfg.s_subgraph)

    def update_actor_and_alpha(self, obs, reward, expl_action):
        self.optimizers.actor.zero_grad()
        self.optimizers.temperature.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            expl_action = None
            distribution, actor_Q, action, side_loss = self.forwarder.forward(self.model, obs, State, self.device,
                                                                              expl_action=expl_action, policy_opt=True,
                                                                              grad=True)

            log_prob = distribution.log_prob(action)
            actor_loss = torch.tensor([0.0], device=actor_Q[0].device)
            alpha_loss = torch.tensor([0.0], device=actor_Q[0].device)
            _log_prob, sg_entropy = [], []
            for i, sz in enumerate(self.cfg.s_subgraph):
                ret = get_joint_sg_logprobs_edges(log_prob, distribution.scale, obs, i, sz)
                _log_prob.append(ret[0])
                sg_entropy.append(ret[1])

                loss = (self.model.alpha[i].detach() * _log_prob[i] - actor_Q[i]).mean()
                actor_loss = actor_loss + loss

            actor_loss = actor_loss / len(self.cfg.s_subgraph) + self.cfg.side_loss_weight * side_loss

            min_entropy = (self.cfg.entropy_range[1] - self.cfg.entropy_range[0]) * ((1.5 - reward[-1]) / 1.5) + \
                          self.cfg.entropy_range[0]
            min_entropy = torch.ones_like(min_entropy)

            for i, sz in enumerate(self.cfg.s_subgraph):
                min_entropy = min_entropy.to(self.model.alpha[i].device).squeeze()
                entropy = sg_entropy[i].detach() if self.cfg.use_closed_form_entropy else -_log_prob[i].detach()
                alpha_loss = alpha_loss + (
                        self.model.alpha[i] * (entropy - (self.cfg.s_subgraph[i] * min_entropy))).mean()

            alpha_loss = alpha_loss / len(self.cfg.s_subgraph)

        self.scalers.actor.scale(actor_loss).backward()
        self.scalers.actor.scale(alpha_loss).backward()
        self.scalers.actor.step(self.optimizers.actor)
        self.scalers.actor.step(self.optimizers.temperature)
        self.scalers.actor.update()

        return actor_loss.item(), alpha_loss.item(), min_entropy, distribution.loc.mean().item()

    def _step(self, step):
        actor_loss, alpha_loss, min_entropy, loc_mean = None, None, None, None

        (obs, action, reward, next_obs, done), sample_idx = self.memory.sample()

        critic_loss, mean_reward = self.update_critic(obs, next_obs, action, reward, not done)
        self.memory.report_sample_loss(critic_loss + mean_reward, sample_idx)
        avg = self.mov_sum_losses.critic.apply(critic_loss)
        if avg is not None:
            self.optimizers.critic_shed.step(avg)
        wandb.log({"loss/critic": critic_loss}, step=self.global_counter)

        if self.cfg.actor_update_after < step and step % self.cfg.actor_update_frequency == 0:
            actor_loss, alpha_loss, min_entropy, loc_mean = self.update_actor_and_alpha(obs, reward, action)
            avg = self.mov_sum_losses.actor.apply(actor_loss)
            if avg is not None:
                self.optimizers.actor_shed.step(avg)
            avg = self.mov_sum_losses.temperature.apply(alpha_loss)
            if avg is not None:
                self.optimizers.temp_shed.step(avg)
            wandb.log({"loss/actor": actor_loss}, step=self.global_counter)
            wandb.log({"loss/alpha": alpha_loss}, step=self.global_counter)

        if step % self.cfg.post_stats_frequency == 0:
            if min_entropy != "nl":
                wandb.log({"min_entropy": min_entropy}, step=self.global_counter)
            wandb.log({"mov_avg/critic": self.mov_sum_losses.critic.avg}, step=self.global_counter)
            wandb.log({"mov_avg/actor": self.mov_sum_losses.actor.avg}, step=self.global_counter)
            wandb.log({"mov_avg/temperature": self.mov_sum_losses.temperature.avg}, step=self.global_counter)
            wandb.log({"lr/critic": self.optimizers.critic_shed.optimizer.param_groups[0]['lr']},
                      step=self.global_counter)
            wandb.log({"lr/actor": self.optimizers.actor_shed.optimizer.param_groups[0]['lr']},
                      step=self.global_counter)
            wandb.log({"lr/temperature": self.optimizers.temp_shed.optimizer.param_groups[0]['lr']},
                      step=self.global_counter)

        self.global_counter = self.global_counter + 1

        if step % self.cfg.critic_target_update_frequency == 0:
            soft_update_params(self.model.critic, self.model.critic_tgt, self.cfg.critic_tau)
            soft_update_params(self.model.fe_ext, self.model.fe_ext_tgt, self.cfg.critic_tau)

        return critic_loss, actor_loss, alpha_loss, loc_mean

    def train_until_finished(self):
        while self.global_count.value() <= self.cfg.T_max + self.cfg.mem_size:
            self.model_mtx.acquire()
            try:
                stats = [[], [], [], []]
                for i in range(self.cfg.n_updates_per_step):
                    _stats = self._step(self.global_count.value())
                    [s.append(_s) for s, _s in zip(stats, _stats)]
                for j in range(len(stats)):
                    if any([_s is None for _s in stats[j]]):
                        stats[j] = "nl"
                    else:
                        stats[j] = round(sum(stats[j]) / self.cfg.n_updates_per_step, 5)

                if self.cfg.verbose:
                    print(
                        f"step: {self.global_count.value()}; mean_loc: {stats[-1]}; n_explorer_steps {self.memory.push_count}",
                        end="")
                    print(f"; cl: {stats[0]}; acl: {stats[1]}; al: {stats[3]}")
            finally:
                self.model_mtx.release()
                self.global_count.increment()
                self.memory.reset_push_count()
            if self.global_count.value() % self.cfg.validatoin_freq == 0:
                self.validate()
        torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, "last_checkpoint_agent.pth"))

    # Acts and trains model
    def train_and_explore(self, rn):
        self.global_count.reset()

        set_seed_everywhere(rn)
        wandb.config.random_seed = rn
        if self.cfg.verbose:
            print('###### start training ######')
            print('Running on device: ', self.device)
            print('found ', self.train_dset.length, " training data patches")
            print('found ', self.val_dset.length, "validation data patches")
            print('training with seed: ' + str(rn))
        explorers = []
        for i in range(self.cfg.n_explorers):
            explorers.append(threading.Thread(target=self.explore))
        [explorer.start() for explorer in explorers]

        self.memory.is_full_event.wait()
        trainer = threading.Thread(target=self.train_until_finished)
        trainer.start()

        trainer.join()
        self.global_count.set(self.cfg.T_max + self.cfg.mem_size + 4)
        [explorer.join() for explorer in explorers]
        self.memory.clear()
        del self.memory
        # torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, "last_checkpoint_agent.pth"))
        if self.cfg.verbose:
            print('\n\n###### training finished ######')
        return

    def explore(self):
        env = MulticutEmbeddingsEnv(self.cfg, self.device)
        tau = 1
        while self.global_count.value() <= self.cfg.T_max + self.cfg.mem_size:
            dloader = iter(DataLoader(self.train_dset, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True,
                                      num_workers=0))
            for iteration in range((len(self.train_dset) // self.cfg.batch_size) * self.cfg.data_update_frequency):
                if iteration % self.cfg.data_update_frequency == 0:
                    update_env_data(env, dloader, self.train_dset, self.device,
                                    with_gt_edges="sub_graph_dice" in self.cfg.reward_function)
                env.reset()
                state = env.get_state()
                done = False
                next_state = None
                while not done:
                    if next_state is not None:
                        state = next_state
                    if not self.memory.is_full():
                        action = torch.rand((env.edge_ids.shape[-1], 1), device=self.device)
                    else:
                        self.model_mtx.acquire()
                        try:
                            distr, _, action, _, _ = self.forwarder.forward(self.model, state, State, self.device,
                                                                            grad=False)
                        finally:
                            self.model_mtx.release()

                    reward, done = env.execute_action(action, tau=max(0, tau))
                    next_state = env.get_state()
                    self.memory.push(state_to_cpu(state, State), action, reward, state_to_cpu(next_state, State), done)
                if self.global_count.value() > self.cfg.T_max + self.cfg.mem_size:
                    break
        return
