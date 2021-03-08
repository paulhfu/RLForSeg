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

from environments.multicut import MulticutEmbeddingsEnv
from data.spg_dset import SpgDset
from models.agent_model import Agent
from models.feature_extractor import FeExtractor
from utils.exploration_functions import RunningAverage
from utils.general import soft_update_params, set_seed_everywhere, cluster_embeddings, pca_project, random_label_cmap
from utils.replay_memory import TransitionData_ts
from utils.graphs import get_joint_sg_logprobs_edges
from utils.distances import CosineDistance, L2Distance
from utils.matching import matching
from utils.yaml_conv_parser import dict_to_attrdict
from utils.training_helpers import update_env_data, supervised_policy_pretraining, agent_forward, state_to_cpu
# from timeit import default_timer as timer




class AgentSacTrainer(object):

    def __init__(self, cfg, global_count):
        super(AgentSacTrainer, self).__init__()

        self.cfg = cfg
        self.global_count = global_count
        self.memory = TransitionData_ts(capacity=self.cfg.t_max)
        self.best_val_reward = -np.inf

    def validate_and_compare_to_clustering(self, model, env, dset, device):
        """validates the prediction against the method of clustering the embedding space"""

        if self.cfg.verbose:
            print("\n\n###### start validate ######", end='')
        model.eval()
        n_examples = len(dset)
        taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        rl_scores, keys = [], None
        ex_raws, ex_sps, ex_gts, ex_mc_gts, ex_embeds, ex_rl = [], [], [], [], [], []
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
            gt_seg = env.gt_seg[0].cpu().numpy()
            gt_mc = cm.prism(env.gt_soln[0].cpu()/env.gt_soln[0].max().item()) if env.gt_edge_weights is not None else torch.zeros(env.raw.shape[-2:])
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
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "best_checkpoint_agent.pth"))
        if self.cfg.verbose:
            print("\n###### finish validate ######\n", end='')

        for i in range(n_examples):
            fig, axs = plt.subplots(2, 3, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
            axs[0, 0].imshow(ex_gts[i], cmap=random_label_cmap(), interpolation="none")
            axs[0, 0].set_title('gt')
            axs[0, 0].axis('off')
            axs[0, 1].imshow(ex_raws[i])
            axs[0, 1].set_title('raw image')
            axs[0, 1].axis('off')
            axs[0, 2].imshow(ex_sps[i], cmap=random_label_cmap(), interpolation="none")
            axs[0, 2].set_title('superpixels')
            axs[0, 2].axis('off')
            axs[1, 0].imshow(ex_embeds[i])
            axs[1, 0].set_title('pc proj 1-3')
            axs[1, 0].axis('off')
            axs[1, 1].imshow(ex_mc_gts[i], cmap=random_label_cmap(), interpolation="none")
            axs[1, 1].set_title('sp gt')
            axs[1, 1].axis('off')
            axs[1, 2].imshow(ex_rl[i], cmap=random_label_cmap(), interpolation="none")
            axs[1, 2].set_title('prediction')
            axs[1, 2].axis('off')
            wandb.log({"validation/samples": [wandb.Image(fig, caption="sample images")]})
            plt.close('all')

    def update_critic(self, obs, action, reward, env, model, scaler, optimizers):
        optimizers.critic.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            current_Q1, current_Q2, side_loss = agent_forward(env, model, obs, actions=action)

            critic_loss = torch.tensor([0.0], device=current_Q1[0].device)
            mean_reward = 0

            for i, sz in enumerate(self.cfg.s_subgraph):
                target_Q = reward[i]
                target_Q = target_Q.detach()

                critic_loss = critic_loss + (F.mse_loss(current_Q1[i], target_Q) + F.mse_loss(current_Q2[i], target_Q))
                mean_reward += reward[i].mean()
            critic_loss = critic_loss / len(self.cfg.s_subgraph) + self.cfg.side_loss_weight * side_loss

        scaler.scale(critic_loss).backward()
        scaler.step(optimizers.critic)
        scaler.update()

        return round(critic_loss.item(), 5), mean_reward / len(self.cfg.s_subgraph)

    def update_actor_and_alpha(self, obs, reward, env, model, scaler, optimizers):
        optimizers.actor.zero_grad()
        optimizers.temperature.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            distribution, actor_Q1, actor_Q2, action, side_loss = agent_forward(env, model, obs, policy_opt=True)

            log_prob = distribution.log_prob(action)
            actor_loss = torch.tensor([0.0], device=actor_Q1[0].device)
            alpha_loss = torch.tensor([0.0], device=actor_Q1[0].device)
            _log_prob, sg_entropy = [], []
            for i, sz in enumerate(self.cfg.s_subgraph):
                actor_Q = torch.min(actor_Q1[i], actor_Q2[i])
                ret = get_joint_sg_logprobs_edges(log_prob, distribution.scale, obs, i, sz)
                _log_prob.append(ret[0])
                sg_entropy.append(ret[1])

                loss = (model.alpha[i].detach() * _log_prob[i] - actor_Q).mean()
                actor_loss = actor_loss + loss

            actor_loss = actor_loss / len(self.cfg.s_subgraph) + self.cfg.side_loss_weight * side_loss

            min_entropy = (self.cfg.entropy_range[1] - self.cfg.entropy_range[0]) * ((1.5 - reward[-1]) / 1.5) + \
                          self.cfg.entropy_range[0]

            for i, sz in enumerate(self.cfg.s_subgraph):
                min_entropy = min_entropy.to(model.alpha[i].device).squeeze()
                entropy = sg_entropy[i].detach() if self.cfg.use_closed_form_entropy else -_log_prob[i].detach()
                alpha_loss = alpha_loss + (model.alpha[i] * (entropy - (self.cfg.s_subgraph[i] * min_entropy))).mean()

            alpha_loss = alpha_loss / len(self.cfg.s_subgraph)

        scaler.scale(actor_loss).backward()
        scaler.scale(alpha_loss).backward()
        scaler.step(optimizers.actor)
        scaler.step(optimizers.temperature)
        scaler.update()

        return round(actor_loss.item(), 5), round(alpha_loss.item(), 5), min_entropy

    def _step(self, replay_buffer, optimizers, mov_sum_loss, env, model, scalers, step, post_data):
        actor_loss, alpha_loss, min_entropy = "nl", "nl", "nl"

        (obs, action, reward), sample_idx = replay_buffer.sample()

        critic_loss, mean_reward = self.update_critic(obs, action, reward, env, model, scalers.critic, optimizers)
        replay_buffer.report_sample_loss(critic_loss + mean_reward, sample_idx)
        mov_sum_loss.critic.apply(critic_loss)
        optimizers.critic_shed.step(mov_sum_loss.critic.avg)
        wandb.log({"loss/critic": critic_loss})

        if self.cfg.actor_update_after < step and step % self.cfg.actor_update_frequency == 0:
            actor_loss, alpha_loss, min_entropy = self.update_actor_and_alpha(obs, reward, env, model, scalers.actor,
                                                                              optimizers)
            mov_sum_loss.actor.apply(actor_loss)
            mov_sum_loss.temperature.apply(alpha_loss)
            optimizers.actor_shed.step(mov_sum_loss.actor.avg)
            optimizers.temp_shed.step(mov_sum_loss.actor.avg)
            wandb.log({"loss/actor": actor_loss})
            wandb.log({"loss/alpha": alpha_loss})

        if step % self.cfg.post_stats_frequency == 0:
            if min_entropy != "nl":
                wandb.log({"min_entropy": min_entropy})
            wandb.log({"mov_avg/critic": mov_sum_loss.critic.avg})
            wandb.log({"mov_avg/actor": mov_sum_loss.actor.avg})
            wandb.log({"mov_avg/temperature": mov_sum_loss.temperature.avg})
            wandb.log({"lr/critic": optimizers.critic_shed.optimizer.param_groups[0]['lr']})
            wandb.log({"lr/actor": optimizers.actor_shed.optimizer.param_groups[0]['lr']})
            wandb.log({"lr/temperature": optimizers.temp_shed.optimizer.param_groups[0]['lr']})

        if step % self.cfg.critic_target_update_frequency == 0:
            soft_update_params(model.critic, model.critic_tgt, self.cfg.critic_tau)

        return critic_loss, actor_loss, alpha_loss

    # Acts and trains model
    def train(self, rank, return_dict, rn):
        self.global_count.reset()

        set_seed_everywhere(rn)
        wandb.config.random_seed = rn
        if self.cfg.verbose:
            print('training with seed: ' + str(rn))
        score = self.train_step(rank)
        return_dict['score'] = score
        del self.memory
        return

    def train_step(self, rank):
        assert torch.cuda.device_count() == 1
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.set_default_tensor_type(torch.FloatTensor)
        if self.cfg.distance == 'cosine':
            self.distance = CosineDistance()
        else:
            self.distance = L2Distance()

        fe_ext = FeExtractor(dict_to_attrdict(self.cfg.backbone), self.distance, device)
        fe_ext.embed_model.load_state_dict(torch.load(self.cfg.fe_model_name))
        fe_ext.cuda(device)

        env = MulticutEmbeddingsEnv(fe_ext, self.cfg, device)

        model = Agent(self.cfg, env.State, self.distance, device)
        wandb.watch(model)
        model.cuda(device)

        MovSumLosses = namedtuple('mov_avg_losses', ('actor', 'critic', 'temperature'))
        Scalers = namedtuple('Scalers', ('critic', 'actor'))
        OptimizerContainer = namedtuple('OptimizerContainer',
                                        ('actor', 'critic', 'temperature', 'actor_shed', 'critic_shed', 'temp_shed'))
        actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=self.cfg.actor_lr)
        critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=self.cfg.critic_lr)
        temp_optimizer = torch.optim.Adam([model.log_alpha], lr=self.cfg.alpha_lr)

        lr_sched_cfg = dict_to_attrdict(self.cfg.lr_sched)
        bw = lr_sched_cfg.mov_avg_bandwidth
        off = lr_sched_cfg.mov_avg_offset
        weights = np.linspace(lr_sched_cfg.weight_range[0], lr_sched_cfg.weight_range[1], bw)
        weights = weights / weights.sum()  # make them sum up to one
        shed = lr_sched_cfg.torch_sched

        mov_sum_losses = MovSumLosses(RunningAverage(weights, band_width=bw, offset=off),
                                      RunningAverage(weights, band_width=bw, offset=off),
                                      RunningAverage(weights, band_width=bw, offset=off))
        optimizers = OptimizerContainer(actor_optimizer, critic_optimizer, temp_optimizer,
                                        *[ReduceLROnPlateau(opt, patience=shed.patience,
                                                            threshold=shed.threshold, min_lr=shed.min_lr,
                                                            factor=shed.factor) for opt in
                                          (actor_optimizer, critic_optimizer, temp_optimizer)])
        scalers = Scalers(torch.cuda.amp.GradScaler(), torch.cuda.amp.GradScaler())

        if self.cfg.agent_model_name != "":
            model.load_state_dict(torch.load(self.cfg.agent_model_name))
        if "policy_warmup" in self.cfg and rank == 0 and self.cfg.agent_model_name == "":
            supervised_policy_pretraining(model, env, self.cfg, device=device)
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "sv_pretrained_policy_agent.pth"))

        # finished with prepping
        for param in fe_ext.parameters():
            param.requires_grad = False

        dset = SpgDset(self.cfg.data_dir, dict_to_attrdict(self.cfg.patch_manager), max(self.cfg.s_subgraph))
        val_dset = SpgDset(self.cfg.val_data_dir, dict_to_attrdict(self.cfg.patch_manager), max(self.cfg.s_subgraph))
        tau = 1

        if self.cfg.verbose:
            print('###### start training ######')
            print('Running on device: ', device)
            print('found ', dset.length, " training data patches")
            print('found ', val_dset.length, "validation data patches")

        while self.global_count.value() <= self.cfg.T_max + self.cfg.t_max:
            dloader = DataLoader(dset, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True, num_workers=0)
            for iteration in range(len(dset) * self.cfg.data_update_frequency):

                post_stats = True if (self.global_count.value()) % self.cfg.post_stats_frequency == 0 else False
                post_images = True if (self.global_count.value()) % (self.cfg.post_stats_frequency * 5) == 0 else False
                post_stats &= self.memory.is_full()

                if iteration % self.cfg.data_update_frequency == 0:
                    update_env_data(env, dloader, self.cfg, device, with_gt_edges="sub_graph_dice" in self.cfg.reward_function)
                env.reset()
                state = env.get_state()
                distr = None

                if not self.memory.is_full():
                    action = torch.rand((env.edge_ids.shape[-1], 1), device=device)
                else:
                    distr, _, _, action, _, _ = agent_forward(env, model, state, grad=False, post_data=post_stats)

                logg_dict = {}
                loc_mean = round(distr.loc.mean().item(), 5) if distr is not None else "nl"
                if post_stats:
                    for i in range(len(self.cfg.s_subgraph)):
                        logg_dict['alpha_' + str(i)] = model.alpha[i].item()
                    if distr is not None:
                        logg_dict['mean_loc'] = loc_mean
                        logg_dict['mean_scale'] = distr.scale.mean().item()
                        logg_dict['tau'] = max(0, tau)

                if self.cfg.verbose:
                    print(f"\nstep: {self.global_count.value()}; mean_loc: {loc_mean}; tau: {round(max(0, tau), 5)}",
                          end='')
                if self.memory.is_full():
                    for i in range(self.cfg.n_updates_per_step):
                        critic_loss, actor_loss, alpha_loss = self._step(self.memory, optimizers, mov_sum_losses, env,
                                                                         model, scalers, self.global_count.value(),
                                                                         post_stats)
                    if self.cfg.verbose:
                        print(f"; cl: {critic_loss}; acl: {actor_loss}; al: {alpha_loss}", end='')
                    tau -= 0.0001

                reward = env.execute_action(action, logg_dict, post_stats=post_stats, tau=max(0, tau), post_images=post_images)

                self.memory.push(state_to_cpu(state, env.State), action, reward)

                self.global_count.increment()
                if self.global_count.value() % self.cfg.validatoin_freq == 0:
                    self.validate_and_compare_to_clustering(model, env, val_dset, device)
                if self.global_count.value() > self.cfg.T_max + self.cfg.t_max:
                    break

        self.memory.clear()
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, "last_checkpoint_agent.pth"))
        if self.cfg.verbose:
            print('\n\n###### training finished ######')

        return sum(env.acc_reward) / len(env.acc_reward)
