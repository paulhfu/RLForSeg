import os
# os.environ['OMP_NUM_THREADS'] = '10'
# os.environ['MKL_NUM_THREADS'] = '10'
import torch
import wandb
import yaml
from utils.general import Counter
from agents.sac import AgentSacTrainer
from agents.sa_obj_lvl_rew import AgentSaTrainerObjLvlReward
from agents.sac_obj_lvl_rew import AgentSacTrainerObjLvlReward
from agents.a2c import AgentA2CTrainer
import multiprocessing as mp

def main(cfg):
    if cfg.verbose:
        print(yaml.dump(dict(cfg), sort_keys=False, default_flow_style=False))
    global_count = Counter()  # Global shared counter
    if cfg.agent == "sac":
        if "s_subgraph" in cfg:
            trainer = AgentSacTrainer(cfg, global_count)
        else:
            trainer = AgentSacTrainerObjLvlReward(cfg, global_count)
    elif cfg.agent == "sa":
        trainer = AgentSaTrainerObjLvlReward(cfg, global_count)
    elif cfg.agent == "a2c":
        trainer = AgentA2CTrainer(cfg, global_count)
    else:
        assert False, "agent not known, check config"
    seed = torch.randint(0, 2 ** 32, torch.Size([1])).item()
    trainer.train_and_explore(seed)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    wandb.init(project="dbg", entity="aule", config="conf/color_circles_configs.yaml")
    main(wandb.config)
