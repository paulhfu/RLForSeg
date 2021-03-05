import os
# os.environ['OMP_NUM_THREADS'] = '10'
# os.environ['MKL_NUM_THREADS'] = '10'
import torch
import wandb
import yaml
from utils.general import Counter
from agents.sac import AgentSacTrainer
# wandb.init(project="RL for Segmentation", entity="aule", config="conf/default_configs.yaml")
wandb.init(project="RL for Segmentation Debuggin Session", entity="aule", config="conf/dbg_configs.yaml")

def main(cfg):
    if cfg.verbose:
        print(yaml.dump(dict(cfg), sort_keys=False, default_flow_style=False))
    global_count = Counter()  # Global shared counter
    trainer = AgentSacTrainer(cfg, global_count)
    return_dict = dict()
    rn = torch.randint(0, 2 ** 32, torch.Size([1])).item()
    trainer.train(0, return_dict, rn)

if __name__ == '__main__':
    main(wandb.config)
