import os
import yaml
import portalocker
import torch
import numpy as np
import torch.distributed as dist
from shutil import copyfile
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from collections import namedtuple
import matplotlib.pyplot as plt

from environments.multicut import MulticutEmbeddingsEnv
from data.spg_dset import SpgDset
from models.agent_model import Agent
from models.feature_extractor import FeExtractor
from utils.exploration_functions import RunningAverage
from utils.general import adjust_learning_rate, soft_update_params, set_seed_everywhere
from utils.replay_memory import TransitionData_ts
from utils.graphs import get_joint_sg_logprobs_edges
from utils.distances import CosineDistance, L2Distance
from utils.training_helpers import update_env_data, pretrain_embeddings, supervised_policy_pretraining, validate, \
    agent_forward, cleanup, state_to_cpu, update_rt_vars


class AgentSacTrainer(object):

    def __init__(self, cfg, global_count, global_writer_loss_count,
                 env_count_val, env_count_train, action_stats_count, global_writer_count, save_dir):
        super(AgentSacTrainer, self).__init__()

        self.cfg = cfg
        self.global_count = global_count
        self.global_writer_loss_count = global_writer_loss_count
        self.global_writer_env_count_val = env_count_val
        self.global_writer_env_count_train = env_count_train
        self.action_stats_count = action_stats_count
        self.global_writer_count = global_writer_count
        self.memory = TransitionData_ts(capacity=self.cfg.trainer.t_max)
        self.save_dir = save_dir

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

    def validation_round(self, dset, model, env, device):
        dloader = DataLoader(dset, batch_size=1, shuffle=True, pin_memory=True, num_workers=0)
        for iteration in range(len(dset)):
            update_env_data(env, dloader, self.cfg, device,
                            with_gt_edges="sub_graph_dice" in self.cfg.sac.reward_function)
            env.reset()
            state = env.get_state()

            distr, _, _, _, _, _ = agent_forward(env, model, state, self.global_count, grad=False,
                                                 post_input=False, post_model=False)
            action = torch.sigmoid(distr.loc)
            env.execute_action(action, None, post_images=True, tau=0.0, train=False)

    def update_critic(self, obs, action, reward, env, model, optimizers):
        current_Q1, current_Q2, side_loss = agent_forward(env, model, obs, self.global_count, actions=action)

        critic_loss = torch.tensor([0.0], device=current_Q1[0].device)
        mean_reward = 0

        for i, sz in enumerate(self.cfg.sac.s_subgraph):
            target_Q = reward[i]
            target_Q = target_Q.detach()

            critic_loss = critic_loss + (F.mse_loss(current_Q1[i], target_Q) + F.mse_loss(current_Q2[i], target_Q))
            mean_reward += reward[i].mean()
        critic_loss = critic_loss / len(self.cfg.sac.s_subgraph) + side_loss
        optimizers.critic.zero_grad()
        critic_loss.backward()
        optimizers.critic.step()

        return critic_loss.item(), mean_reward / len(self.cfg.sac.s_subgraph)

    def update_actor_and_alpha(self, obs, reward, env, model, optimizers):
        distribution, actor_Q1, actor_Q2, action, side_loss = agent_forward(env, model, obs, self.global_count, policy_opt=True)

        log_prob = distribution.log_prob(action)
        actor_loss = torch.tensor([0.0], device=actor_Q1[0].device)
        alpha_loss = torch.tensor([0.0], device=actor_Q1[0].device)
        _log_prob, sg_entropy = [], []
        for i, sz in enumerate(self.cfg.sac.s_subgraph):
            actor_Q = torch.min(actor_Q1[i], actor_Q2[i])
            ret = get_joint_sg_logprobs_edges(log_prob, distribution.scale, obs, i, sz)
            _log_prob.append(ret[0])
            sg_entropy.append(ret[1])

            loss = (model.module.alpha[i].detach() * _log_prob[i] - actor_Q).mean()
            actor_loss = actor_loss + loss

        actor_loss = actor_loss / len(self.cfg.sac.s_subgraph) + side_loss
        optimizers.actor.zero_grad()
        actor_loss.backward()
        optimizers.actor.step()

        min_entropy = (self.cfg.sac.entropy_range[1] - self.cfg.sac.entropy_range[0]) * ((1.5 - reward[-1]) / 1.5) \
                      + self.cfg.sac.entropy_range[0]

        for i, sz in enumerate(self.cfg.sac.s_subgraph):
            # min_entropy = min_entropy.to(model.module.alpha[i].device).squeeze()
            entropy = sg_entropy[i].detach() if self.cfg.sac.use_closed_form_entropy else -_log_prob[i].detach()
            alpha_loss = alpha_loss + (model.module.alpha[i] * (entropy - (self.cfg.sac.s_subgraph[i]))).mean()

        alpha_loss = alpha_loss / len(self.cfg.sac.s_subgraph)
        optimizers.temperature.zero_grad()
        alpha_loss.backward()
        optimizers.temperature.step()

        return actor_loss.item(), alpha_loss.item(), min_entropy

    def _step(self, replay_buffer, optimizers, mov_sum_loss, env, model, step, writer=None):
        critic_loss, actor_loss, alpha_loss = "nl", "nl", "nl"

        (obs, action, reward), sample_idx = replay_buffer.sample()

        if step % self.cfg.sac.critic_target_update_frequency == 0 or self.cfg.sac.actor_update_after < step:
            critic_loss, mean_reward = self.update_critic(obs, action, reward, env, model, optimizers)
            replay_buffer.report_sample_loss(critic_loss + mean_reward, sample_idx)
            mov_sum_loss.critic.apply(critic_loss)

        if self.cfg.sac.actor_update_after < step and step % self.cfg.sac.actor_update_frequency == 0:
            actor_loss, alpha_loss, min_entropy = self.update_actor_and_alpha(obs, reward, env, model, optimizers)
            mov_sum_loss.actor.apply(actor_loss)
            mov_sum_loss.temperature.apply(alpha_loss)
            writer.add_scalar("min_entropy", min_entropy, self.global_writer_loss_count.value())

        if self.global_writer_loss_count.value() > self.cfg.trainer.lr_sched.mov_avg_bandwidth \
                and self.global_writer_loss_count.value() % self.cfg.trainer.lr_sched.step_frequency == 0:
            optimizers.critic_shed.step(mov_sum_loss.critic.avg)
            if self.cfg.sac.actor_update_after < step:
                optimizers.actor_shed.step(mov_sum_loss.actor.avg)
                optimizers.temp_shed.step(mov_sum_loss.actor.avg)
            writer.add_scalar("mov_sum/critic", mov_sum_loss.critic.avg, self.global_writer_loss_count.value())
            writer.add_scalar("mov_sum/actor", mov_sum_loss.actor.avg, self.global_writer_loss_count.value())
            writer.add_scalar("mov_sum/temperature", mov_sum_loss.temperature.avg, self.global_writer_loss_count.value())
            writer.add_scalar("lr/critic", optimizers.critic_shed.optimizer.param_groups[0]['lr'], self.global_writer_loss_count.value())
            writer.add_scalar("lr/actor", optimizers.actor_shed.optimizer.param_groups[0]['lr'], self.global_writer_loss_count.value())
            writer.add_scalar("lr/temperature", optimizers.temp_shed.optimizer.param_groups[0]['lr'], self.global_writer_loss_count.value())

        if step % self.cfg.sac.critic_target_update_frequency == 0:
            soft_update_params(model.module.critic, model.module.critic_tgt, self.cfg.sac.critic_tau)

        self.global_writer_loss_count.increment()
        return critic_loss, actor_loss, alpha_loss

    # Acts and trains model
    def train(self, rank, return_dict, rn):

        self.log_dir = os.path.join(self.save_dir, 'logs', '_' + str(rn))
        writer = None
        if rank == 0:
            writer = SummaryWriter(logdir=self.log_dir)
            writer.add_text("conf", yaml.dump(self.cfg, sort_keys=False, default_flow_style=False))
            # writer.add_text("conf", self.cfg.pretty())
            copyfile(os.path.join(self.save_dir, 'runtime_cfg.yaml'),
                     os.path.join(self.log_dir, 'runtime_cfg.yaml'))

            self.global_count.reset()
            self.global_writer_loss_count.reset()
            self.global_writer_env_count_val.reset()
            self.global_writer_env_count_train.reset()
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
        device = torch.device("cuda:" + str(rank // self.cfg.gen.n_processes_per_gpu))
        print('Running on device: ', device)
        torch.cuda.set_device(device)
        torch.set_default_tensor_type(torch.FloatTensor)
        self.setup(rank, self.cfg.gen.n_processes_per_gpu * self.cfg.gen.n_gpu)
        # cosine distance (input should always be normalized)
        if self.cfg.fe.distance == 'cosine':
            self.distance = CosineDistance()
        else:
            self.distance = L2Distance()

        fe_ext = FeExtractor(self.cfg.fe.backbone, self.distance, device, writer)
        fe_ext.cuda(device)
        env = MulticutEmbeddingsEnv(fe_ext, self.cfg, device, writer=writer,
                                    writer_counter_val=self.global_writer_env_count_val,
                                    writer_counter_train=self.global_writer_env_count_train)

        model = Agent(self.cfg, env.State, self.distance, device, writer=writer)
        model.cuda(device)
        # Create shared network
        shared_model = DDP(model, device_ids=[device], find_unused_parameters=True)
        MovSumLosses = namedtuple('mov_avg_losses', ('actor', 'critic', 'temperature'))
        OptimizerContainer = namedtuple('OptimizerContainer', ('actor', 'critic', 'temperature',
                                                               'actor_shed', 'critic_shed', 'temp_shed'))
        actor_optimizer = torch.optim.Adam(shared_model.module.actor.parameters(),
                                           lr=self.cfg.sac.actor_lr)
        critic_optimizer = torch.optim.Adam(shared_model.module.critic.parameters(),
                                            lr=self.cfg.sac.critic_lr)
        temp_optimizer = torch.optim.Adam([shared_model.module.log_alpha],
                                          lr=self.cfg.sac.alpha_lr)

        bw = self.cfg.trainer.lr_sched.mov_avg_bandwidth
        off = self.cfg.trainer.lr_sched.mov_avg_offset
        weights = np.linspace(self.cfg.trainer.lr_sched.weight_range[0], self.cfg.trainer.lr_sched.weight_range[1], bw)
        weights = weights / weights.sum()  # make them sum up to one
        shed = self.cfg.trainer.lr_sched.torch_sched

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
            shared_model.module.load_state_dict(torch.load(self.cfg.gen.model_name))
            if self.cfg.gen.validation is not None:
                validate(model, env, self.cfg, device)
                return
        elif self.cfg.fe.load_pretrained:
            fe_ext.embed_model.load_state_dict(torch.load(self.cfg.fe.model_name))
        if "policy_warmup" in self.cfg.sac and rank == 0 and not self.cfg.gen.resume:
            supervised_policy_pretraining(shared_model, env, self.cfg, writer, device=device)
            torch.save(shared_model.module.state_dict(), self.cfg.gen.model_name)

        dist.barrier()
        # finished with prepping
        for param in fe_ext.parameters():
            param.requires_grad = False

        dset = SpgDset(self.cfg.gen.data_dir, self.cfg.gen.patch_manager, max(self.cfg.sac.s_subgraph))
        val_dset = SpgDset(self.cfg.gen.val_data_dir, self.cfg.gen.patch_manager, max(self.cfg.sac.s_subgraph))
        tau = 1

        while self.global_count.value() <= self.cfg.trainer.T_max:
            dloader = DataLoader(dset, batch_size=self.cfg.trainer.batch_size, shuffle=True, pin_memory=True, num_workers=0)
            for iteration in range(len(dset) * self.cfg.trainer.data_update_frequency):
                if iteration % self.cfg.trainer.data_update_frequency == 0:
                    update_env_data(env, dloader, self.cfg, device,
                                    with_gt_edges="sub_graph_dice" in self.cfg.sac.reward_function)
                env.reset()
                update_rt_vars(critic_optimizer, actor_optimizer, self.log_dir, self.cfg)
                if rank == 0 and self.cfg.rt_vars.safe_model:
                    if self.cfg.gen.model_name != "":
                        torch.save(shared_model.module.state_dict(), self.cfg.gen.model_name)
                    else:
                        torch.save(shared_model.module.state_dict(), os.path.join(self.log_dir, 'agent_model'))

                state = env.get_state()
                post_stats = True if (self.global_writer_count.value()) % self.cfg.trainer.post_stats_frequency == 0 else False
                post_images = True if (self.global_writer_count.value()) % (self.cfg.trainer.post_stats_frequency * 5) == 0 else False
                post_model = True if (self.global_writer_count.value() + 1) % self.cfg.trainer.post_model_frequency == 0 else False
                post_stats &= self.memory.is_full()
                post_model &= self.memory.is_full()
                distr = None
                if not self.memory.is_full():
                    action = torch.rand((env.edge_ids.shape[-1], self.cfg.sac.n_actions), device=device)
                else:
                    distr, _, _, action, _, _ = agent_forward(env, shared_model, state, self.global_count,
                                                              grad=False, post_input=post_stats,
                                                              post_model=post_model)

                logg_dict = {}
                loc_mean = distr.loc.mean().item() if distr is not None else "nl"
                if post_stats:
                    for i in range(len(self.cfg.sac.s_subgraph)):
                        logg_dict['alpha_' + str(i)] = shared_model.module.alpha[i].item()
                    if distr is not None:
                        logg_dict['mean_loc'] = loc_mean
                        logg_dict['mean_scale'] = distr.scale.mean().item()
                        logg_dict['tau'] = max(0, tau)

                if self.cfg.gen.verbose:
                    print(f"\nstep: {self.global_count.value()}; mean_loc: {loc_mean}; tau: {max(0, tau)}", end='')
                if self.memory.is_full():
                    for i in range(self.cfg.trainer.n_updates_per_step):
                        critic_loss, actor_loss, alpha_loss = self._step(self.memory, optimizers, mov_sum_losses,
                                                                         env, shared_model,
                                                                         self.global_count.value(), writer=writer)
                    if self.cfg.gen.verbose:
                        print(f"; cl: {critic_loss}; acl: {actor_loss}; al: {alpha_loss}", end='')
                    tau -= 0.0001

                reward = env.execute_action(action, logg_dict, post_stats=post_stats, tau=max(0, tau), post_images=post_images)

                self.memory.push(state_to_cpu(state, env.State), action, reward)

                self.global_count.increment()
                if self.global_count.value() % self.cfg.trainer.validatoin_freq == 0:
                    self.validation_round(val_dset, shared_model, env, device)
                if rank == 0:
                    self.global_writer_count.increment()
                if self.global_count.value() > self.cfg.trainer.T_max:
                    break

        dist.barrier()
        if rank == 0:
            self.memory.clear()
            if self.cfg.gen.model_name != "":
                torch.save(shared_model.state_dict(), self.cfg.gen.model_name)
            else:
                torch.save(shared_model.state_dict(), os.path.join(self.log_dir, 'agent_model'))
            print('saved')

        cleanup()
        return sum(env.acc_reward) / len(env.acc_reward)
