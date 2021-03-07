import os
# os.environ['OMP_NUM_THREADS'] = '10'
# os.environ['MKL_NUM_THREADS'] = '10'
import torch
import wandb
import yaml
from utils.general import Counter
from agents.sac import AgentSacTrainer

def main(cfg):
    if cfg.verbose:
        print(yaml.dump(dict(cfg), sort_keys=False, default_flow_style=False))
    global_count = Counter()  # Global shared counter
    trainer = AgentSacTrainer(cfg, global_count)
    return_dict = dict()
    rn = torch.randint(0, 2 ** 32, torch.Size([1])).item()
    trainer.train(0, return_dict, rn)

if __name__ == '__main__':
    wandb.init(project="dbg", entity="aule", config="conf/leptin_configs.yaml")
    main(wandb.config)
