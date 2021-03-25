import matplotlib
matplotlib.use('Agg')
import torch.multiprocessing as mp
from run import main
from utils.yaml_conv_parser import YamlConf
import os
import wandb
import getopt
import sys

if __name__ == '__main__':
    # os.nice(15)
    # project name: "RL for Segmentation"
    base_dir = "/g/kreshuk/hilt/projects/RLForSeg/results/wandb"
    project = "train_sguv_newgnn"
    entity = "rl_segmentation"
    config = "conf/leptin_configs.yaml"

    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:p:e:c:",
                                   ["dir=", "project=", "entity=", "config="])
    except getopt.GetoptError:
        print('help: run.py -d <base_dir> -p <wandb_project> -e <wandb_entity> -c <config_path>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('help: run.py -d <base_dir> -p <wandb_project> -e <wandb_entity> -c <config>')
            sys.exit(0)
        elif opt in ("-d", "--dir"):
            base_dir = arg
        elif opt in ("-p", "--project"):
            project = arg
        elif opt in ("-e", "--entity"):
            entity = arg
        elif opt in ("-c", "--config"):
            config = arg
    # os.nice(15)
    # project name: "RL for Segmentation"
    #print(base_dir, project, entity, config)
    #sys.exit(0)
    wandb.init(dir=base_dir, project=project, entity=entity, config=config)
    main(wandb.config)
