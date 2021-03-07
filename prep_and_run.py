import matplotlib
matplotlib.use('Agg')
import torch.multiprocessing as mp
from run import main
from utils.yaml_conv_parser import YamlConf
import os
import wandb

if __name__ == '__main__':
    # os.nice(15)
    # project name: "RL for Segmentation"
    wandb.init(project="RL for Segmentation", entity="aule", config="conf/leptin_configs.yaml")
    main(wandb.config)
