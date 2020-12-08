import os
# os.environ["OMP_NUM_THREADS"] = "1"
import torch
from utils.general import Counter
from torch import multiprocessing as mp
import hydra
from omegaconf import OmegaConf
import yaml
import numpy as np
from agents.sac import AgentSacTrainer


@hydra.main(config_path="conf")
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

    print(OmegaConf.to_yaml(cfg))

    with open(os.path.join(save_dir, 'conf.txt'), "w") as info:
        info.write(OmegaConf.to_yaml(cfg))

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

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
