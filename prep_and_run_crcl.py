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
    project = "crcl_uv_sac_fe_opt"
    entity = "rl_segmentation"
    config = "conf/color_circles_configs.yaml"
    name = ""
    jobid = ""

    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:p:e:c:n:",
                                   ["dir=", "project=", "entity=", "config=", "name=", "jobid="])
    except getopt.GetoptError:
        print('help: run.py -d <base_dir> -p <wandb_project> -e <wandb_entity> -c <config_path> -n <run_name> -s <slurm jobid>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('help: run.py -d <base_dir> -p <wandb_project> -e <wandb_entity> -c <config> -n <run_name>')
            sys.exit(0)
        elif opt in ("-d", "--dir"):
            base_dir = arg
        elif opt in ("-p", "--project"):
            project = arg
        elif opt in ("-e", "--entity"):
            entity = arg
        elif opt in ("-c", "--config"):
            config = arg
        elif opt in ("-n", "--name"):
            name = arg
        elif opt in ("-s", "--jobid"):
            jobid = arg
    # os.nice(15)
    # project name: "RL for Segmentation"
    #print(base_dir, project, entity, config)
    #sys.exit(0)
    if (name != ""):
        wandb.init(dir=base_dir, project=project, entity=entity, config=config, name=name)
    else:
        wandb.init(dir=base_dir, project=project, entity=entity, config=config)
    wandb.config.update({"SLURM jobid": jobid})
    main(wandb.config)