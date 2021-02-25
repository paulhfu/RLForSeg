import os
# os.environ["OMP_NUM_THREADS"] = "1"
import torch
from utils.general import Counter
from torch import multiprocessing as mp
import yaml
import numpy as np
from agents.sac import AgentSacTrainer
from utils.yaml_conv_parser import YamlConf


def main(cfg):
    # Creating directories.
    save_dir = os.path.join(cfg.gen.base_dir, 'results/sac', cfg.gen.target_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_dir = os.path.join(save_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if os.path.exists(os.path.join(save_dir, 'runtime_cfg.yaml')):
        os.remove(os.path.join(save_dir, 'runtime_cfg.yaml'))

    print(yaml.dump(cfg, sort_keys=False, default_flow_style=False))
    with open(os.path.join(save_dir, 'conf.txt'), "w") as info:
        info.write(yaml.dump(cfg, sort_keys=False, default_flow_style=False))

    rt_cfg_dict = dict(cfg.rt_vars)
    cfg_dict = dict(cfg.gen)
    cfg_dict.update(dict(cfg.fe))
    cfg_dict.update(dict(cfg.sac))
    cfg_dict.update(dict(cfg.trainer))
    for key in rt_cfg_dict:
        if key in cfg_dict and rt_cfg_dict[key] is None:
            rt_cfg_dict[key] = cfg_dict[key]
            cfg.rt_vars[key] = cfg_dict[key]
    with open(os.path.join(save_dir, 'runtime_cfg.yaml'), "w") as info:
        yaml.dump(rt_cfg_dict, info)

    global_count = Counter()  # Global shared counter
    global_writer_count = Counter()
    global_writer_loss_count = Counter()  # Global shared counter
    global_writer_quality_count = Counter()  # Global shared counter
    action_stats_count = Counter()

    trainer = AgentSacTrainer(cfg,
                              global_count,
                              global_writer_loss_count,
                              global_writer_quality_count,
                              action_stats_count=action_stats_count,
                              global_writer_count=global_writer_count,
                              save_dir=save_dir)

    manager = mp.Manager()
    return_dict = manager.dict()
    rns = torch.randint(0, 2 ** 32, torch.Size([4]))
    best_qual = -np.inf
    best_seed = None
    for i, rn in enumerate(rns):
        # Start validation agent
        processes = []
        for rank in range(cfg.gen.n_processes_per_gpu * cfg.gen.n_gpu):
            p = mp.Process(target=trainer.train, args=(rank, return_dict, rn.item()))
            p.start()
            processes.append(p)
        # Clean up
        for p in processes:
            p.join()

        if return_dict['score'] > best_qual:
            best_qual = return_dict['score']
            best_seed = rn.item()

        # sys.stdout = open("/dev/stdout", "w")
        res = 'best seed is: ' + str(best_seed) + " with a qual of: " + str(best_qual)
        print(res)

    with open(os.path.join(save_dir, 'result.txt'), "w") as info:
        info.write(res)

    if cfg.gen.cross_validate_hp or cfg.gen.test_score_only:
        print('Score is: ', return_dict['test_score'])
        return return_dict['test_score']


def no_mp_main(cfg):
    print("dbg2")
    # Creating directories.
    save_dir = os.path.join(cfg.gen.base_dir, 'results/sac', cfg.gen.target_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_dir = os.path.join(save_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if os.path.exists(os.path.join(save_dir, 'runtime_cfg.yaml')):
        os.remove(os.path.join(save_dir, 'runtime_cfg.yaml'))

    print(yaml.dump(cfg, sort_keys=False, default_flow_style=False))
    with open(os.path.join(save_dir, 'conf.txt'), "w") as info:
        info.write(yaml.dump(cfg, sort_keys=False, default_flow_style=False))

    rt_cfg_dict = dict(cfg.rt_vars)
    cfg_dict = dict(cfg.gen)
    cfg_dict.update(dict(cfg.fe))
    cfg_dict.update(dict(cfg.sac))
    cfg_dict.update(dict(cfg.trainer))
    for key in rt_cfg_dict:
        if key in cfg_dict and rt_cfg_dict[key] is None:
            rt_cfg_dict[key] = cfg_dict[key]
            cfg.rt_vars[key] = cfg_dict[key]
    with open(os.path.join(save_dir, 'runtime_cfg.yaml'), "w") as info:
        yaml.dump(rt_cfg_dict, info)

    global_count = Counter()  # Global shared counter
    global_writer_count = Counter()
    global_writer_loss_count = Counter()  # Global shared counter
    env_count_val = Counter()
    env_count_train = Counter()
    action_stats_count = Counter()

    trainer = AgentSacTrainer(cfg,
                              global_count,
                              global_writer_loss_count,
                              env_count_val,
                              env_count_train,
                              action_stats_count=action_stats_count,
                              global_writer_count=global_writer_count,
                              save_dir=save_dir)

    return_dict = dict()
    rn = torch.randint(0, 2 ** 32, torch.Size([1])).item()
    trainer.train(0, return_dict, rn)

if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)
    no_mp_main(YamlConf("conf").cfg)
