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

from environments.multicut_obj_lvl_rew import MulticutEmbeddingsEnv, State
from data.spg_dset import SpgDset
from models.actor_model_obj_lvl import Agent
from models.feature_extractor import FeExtractor
from utils.exploration_functions import RunningAverage
from utils.general import soft_update_params, set_seed_everywhere, cluster_embeddings, pca_project, random_label_cmap
from utils.replay_memory import TransitionData_ts
from utils.distances import CosineDistance, L2Distance
from utils.matching import matching
from utils.yaml_conv_parser import dict_to_attrdict
from utils.training_helpers import update_env_data, supervised_policy_pretraining, state_to_cpu, Forwarder


# from timeit import default_timer as timer


class AgentSaTrainerObjLvlReward(object):

    def __init__(self, cfg, global_count):
        super(AgentSaTrainerObjLvlReward, self).__init__()
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

        self.fe_ext = FeExtractor(dict_to_attrdict(self.cfg.backbone), self.distance, self.device)
        self.fe_ext.embed_model.load_state_dict(torch.load(self.cfg.fe_model_name))
        self.fe_ext.cuda(self.device)

        self.model = Agent(self.cfg, State, self.distance, self.device)
        wandb.watch(self.model)
        self.model.cuda(self.device)
        self.model_mtx = Lock()

        self.optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=self.cfg.actor_lr)

        lr_sched_cfg = dict_to_attrdict(self.cfg.lr_sched)
        bw = lr_sched_cfg.mov_avg_bandwidth
        off = lr_sched_cfg.mov_avg_offset
        weights = np.linspace(lr_sched_cfg.weight_range[0], lr_sched_cfg.weight_range[1], bw)
        weights = weights / weights.sum()  # make them sum up to one
        shed = lr_sched_cfg.torch_sched
        self.shed = ReduceLROnPlateau(self.optimizer, patience=shed.patience, threshold=shed.threshold,
                                      min_lr=shed.min_lr, factor=shed.factor)

        self.mov_sum_loss = RunningAverage(weights, band_width=bw, offset=off)
        self.scaler = torch.cuda.amp.GradScaler()
        self.forwarder = Forwarder()

        if self.cfg.agent_model_name != "":
            self.model.load_state_dict(torch.load(self.cfg.agent_model_name))

        # finished with prepping
        for param in self.fe_ext.parameters():
            param.requires_grad = False

        self.train_dset = SpgDset(self.cfg.data_dir, dict_to_attrdict(self.cfg.patch_manager),
                                  dict_to_attrdict(self.cfg.data_keys))
        self.val_dset = SpgDset(self.cfg.val_data_dir, dict_to_attrdict(self.cfg.patch_manager),
                                dict_to_attrdict(self.cfg.data_keys))

    def validate(self):
        """validates the prediction against the method of clustering the embedding space"""
        env = MulticutEmbeddingsEnv(self.fe_ext, self.cfg, self.device)
        if self.cfg.verbose:
            print("\n\n###### start validate ######", end='')
        self.model.eval()
        n_examples = len(self.val_dset)
        taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        rl_scores, keys = [], None
        ex_raws, ex_sps, ex_gts, ex_mc_gts, ex_embeds, ex_rl = [], [], [], [], [], []
        dloader = iter(DataLoader(self.val_dset, batch_size=1, shuffle=True, pin_memory=True, num_workers=0))
        acc_reward = 0
        for it in range(len(self.val_dset)):
            update_env_data(env, dloader, self.cfg, self.device,
                            with_gt_edges="sub_graph_dice" in self.cfg.reward_function)
            env.reset()
            state = env.get_state()

            self.model_mtx.acquire()
            try:
                distr, _, _ = self.forwarder.forward(self.model, state, State, self.device, grad=False, post_data=False)
            finally:
                self.model_mtx.release()
            action = torch.sigmoid(distr.loc)
            reward, state = env.execute_action(action, None, post_images=True, tau=0.0, train=False)
            acc_reward += reward[-1].item()
            if self.cfg.verbose:
                print(
                    f"\nstep: {it}; mean_loc: {round(distr.loc.mean().item(), 5)}; mean reward: {round(reward[-1].item(), 5)}",
                    end='')

            embeddings = env.embeddings[0].cpu().numpy()
            gt_seg = env.gt_seg[0].cpu().numpy()
            gt_mc = cm.prism(
                env.gt_soln[0].cpu() / env.gt_soln[0].max().item()) if env.gt_edge_weights is not None else torch.zeros(
                env.raw.shape[-2:])
            rl_labels = env.current_soln.cpu().numpy()[0]

            if it < n_examples:
                ex_embeds.append(pca_project(embeddings, n_comps=3))
                ex_raws.append(env.raw[0].cpu().permute(1, 2, 0).squeeze())
                ex_sps.append(env.init_sp_seg[0].cpu())
                ex_mc_gts.append(gt_mc)
                ex_gts.append(gt_seg)
                ex_rl.append(rl_labels)

            _rl_scores = matching(gt_seg, rl_labels, thresh=taus, criterion='iou', report_matches=False)

            if it == 0:
                for tau_it in range(len(_rl_scores)):
                    rl_scores.append(np.array(list(map(float, list(_rl_scores[tau_it]._asdict().values())[1:]))))
                keys = list(_rl_scores[0]._asdict().keys())[1:]
            else:
                for tau_it in range(len(_rl_scores)):
                    rl_scores[tau_it] += np.array(list(map(float, list(_rl_scores[tau_it]._asdict().values())[1:])))

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

        wandb.log({"validation/metrics": [wandb.Image(fig, caption="metrics")]})
        wandb.log({"validation_reward": acc_reward})
        plt.close('all')
        if acc_reward > self.best_val_reward:
            self.best_val_reward = acc_reward
            wandb.run.summary["validation_reward"] = acc_reward
            torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, "best_checkpoint_agent.pth"))
        if self.cfg.verbose:
            print("\n###### finish validate ######\n", end='')

        for i in range(n_examples):
            fig, axs = plt.subplots(2, 3, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
            axs[0, 0].imshow(ex_gts[i], cmap=random_label_cmap(), interpolation="none")
            axs[0, 0].set_title('gt')
            axs[0, 0].axis('off')
            if ex_raws[i].ndim == 3:
                axs[0, 1].imshow(ex_raws[i][..., 0])
            else:
                axs[0, 1].imshow(ex_raws[i])
            axs[0, 1].set_title('raw image')
            axs[0, 1].axis('off')
            axs[0, 2].imshow(ex_sps[i], cmap=random_label_cmap(), interpolation="none")
            axs[0, 2].set_title('superpixels')
            axs[0, 2].axis('off')
            axs[1, 0].imshow(ex_embeds[i])
            axs[1, 0].set_title('pc proj 1-3', y=-0.15)
            axs[1, 0].axis('off')
            if ex_raws[i].ndim == 3:
                if ex_raws[i].shape[-1] > 1:
                    axs[1, 1].imshow(ex_raws[i][..., 1])
                else:
                    axs[1, 1].imshow(ex_raws[i][..., 0])
            else:
                axs[1, 1].imshow(ex_raws[i])
            axs[1, 1].set_title('sp edge', y=-0.15)
            axs[1, 1].axis('off')
            axs[1, 2].imshow(ex_rl[i], cmap=random_label_cmap(), interpolation="none")
            axs[1, 2].set_title('prediction', y=-0.15)
            axs[1, 2].axis('off')
            wandb.log({"validation/samples": [wandb.Image(fig, caption="sample images")]})
            plt.close('all')

    def update_actor_and_alpha(self, obs, reward, expl_action):
        self.optimizer.zero_grad()
        obj_edge_mask_actor = obs.obj_edge_mask_actor.to(self.device)
        with torch.cuda.amp.autocast(enabled=True):
            distribution, action, side_loss = self.forwarder.forward(self.model, obs, State, self.device,
                                                                     expl_action=expl_action, policy_opt=True)
            p = distribution.cdf(action + (1e-2 / 2)) - distribution.cdf(action - (1e-2 / 2))
            obj_logprob = (p[None] * obj_edge_mask_actor[..., None]).log().sum(1)
            _obj_logprob = (1 - (p[None] * obj_edge_mask_actor[..., None])).log().sum(1)
            actor_loss = (-(obj_logprob * reward[0] + _obj_logprob * (1 - reward[0]))).mean()

            obj_entropy = ((1 / 2 * (1 + (2 * np.pi * distribution.scale ** 2).log()))[None] * obj_edge_mask_actor[
                ..., None]).sum(1).squeeze(1)
            min_entropy = (self.cfg.entropy_range[1] - self.cfg.entropy_range[0]) * (1.0 - reward[0]) + \
                          self.cfg.entropy_range[0]
            min_entropy = min_entropy.to(action.device).squeeze()
            entropy = obj_entropy if self.cfg.use_closed_form_entropy else -obj_logprob
            entropy_reg = ((entropy - min_entropy) ** 2).mean()

            loss = actor_loss + entropy_reg + self.cfg.side_loss_weight * side_loss

        self.scaler.scale(actor_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item(), min_entropy.mean().item(), distribution.loc.mean().item()

    def _step(self, step):
        (obs, action, reward), sample_idx = self.memory.sample()
        action = action.to(self.device)
        for i in range(len(reward)):
            reward[i] = reward[i].to(self.device)

        actor_loss, min_entropy, loc_mean = self.update_actor_and_alpha(obs, reward, action)
        self.mov_sum_loss.apply(actor_loss)
        # self.optimizers.actor_shed.step(self.mov_sum_losses.actor.avg)
        wandb.log({"loss/actor": actor_loss})

        if step % self.cfg.post_stats_frequency == 0:
            if min_entropy != "nl":
                wandb.log({"min_entropy": min_entropy})
            wandb.log({"mov_avg/actor": self.mov_sum_loss.avg})
            wandb.log({"lr/actor": self.shed.optimizer.param_groups[0]['lr']})

        return [actor_loss, loc_mean]

    def train_until_finished(self):
        while self.global_count.value() <= self.cfg.T_max + self.cfg.mem_size:
            self.model_mtx.acquire()
            try:
                stats = [[], []]
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
                    print(f"; acl: {stats[0]}")
            finally:
                self.model_mtx.release()
                self.global_count.increment()
                self.memory.reset_push_count()
            if self.global_count.value() % self.cfg.validatoin_freq == 0:
                self.validate()

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
        torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, "last_checkpoint_agent.pth"))
        if self.cfg.verbose:
            print('\n\n###### training finished ######')
        return

    def explore(self):
        env = MulticutEmbeddingsEnv(self.fe_ext, self.cfg, self.device)
        tau = 1
        while self.global_count.value() <= self.cfg.T_max + self.cfg.mem_size:
            dloader = iter(DataLoader(self.train_dset, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True,
                                 num_workers=0))
            for iteration in range(len(self.train_dset) * self.cfg.data_update_frequency):
                if iteration % self.cfg.data_update_frequency == 0:
                    update_env_data(env, dloader, self.cfg, self.device,
                                    with_gt_edges="sub_graph_dice" in self.cfg.reward_function)
                env.reset()
                state = env.get_state()

                if not self.memory.is_full():
                    action = torch.rand((env.edge_ids.shape[-1], 1), device=self.device)
                else:
                    self.model_mtx.acquire()
                    try:
                        distr, action, _ = self.forwarder.forward(self.model, state, State, self.device, grad=False)
                    finally:
                        self.model_mtx.release()
                reward, state = env.execute_action(action, tau=max(0, tau))
                for i in range(len(reward)):
                    reward[i] = reward[i].cpu()

                self.memory.push(state_to_cpu(state, State), action.cpu(), reward)
                if self.global_count.value() > self.cfg.T_max + self.cfg.mem_size:
                    break
        return
