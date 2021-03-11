import os
# os.environ['OMP_NUM_THREADS'] = '10'
# os.environ['MKL_NUM_THREADS'] = '10'
import torch
import wandb
import yaml
from utils.general import Counter
from agents.sac import AgentSacTrainer
import multiprocessing as mp

def main(cfg):
    if cfg.verbose:
        print(yaml.dump(dict(cfg), sort_keys=False, default_flow_style=False))
    global_count = Counter()  # Global shared counter
    trainer = AgentSacTrainer(cfg, global_count)
    seed = torch.randint(0, 2 ** 32, torch.Size([1])).item()
    trainer.train_and_explore(seed)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    wandb.init(project="dbg", entity="aule", config="conf/tlo_configs.yaml")
    main(wandb.config)
