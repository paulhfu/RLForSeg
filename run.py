import matplotlib
matplotlib.use('Agg')
import getopt
import sys
import wandb
import yaml
from utils.general import Counter
from agents.sac import AgentSacTrainer


def main(cfg):
    if cfg.verbose:
        print(yaml.dump(dict(cfg), sort_keys=False, default_flow_style=False))
    global_count = Counter()  # Global shared counter
    trainer = AgentSacTrainer(cfg, global_count)
    trainer.train_and_explore(cfg.random_seed)


if __name__ == '__main__':
    base_dir = ""
    project = ""
    entity = ""
    config = ""
    name = ""
    jobid = ""

    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:p:e:c:n",
                                   ["dir=", "project=", "entity=", "config=", "name="])
    except getopt.GetoptError:
        print('help: run.py -d <base_dir> -p <wandb_project> -e <wandb_entity> -c <config_path> -n <run_name> -s <slurm jobid>')
        sys.exit(2)
    present_args = 0
    for opt, arg in opts:
        if opt == '-h':
            print('help: run.py -d <base_dir> -p <wandb_project> -e <wandb_entity> -c <config> -n <run_name>')
            sys.exit(0)
        elif opt in ("-d", "--dir"):
            base_dir = arg
            present_args += 1
        elif opt in ("-p", "--project"):
            project = arg
            present_args += 1
        elif opt in ("-e", "--entity"):
            entity = arg
            present_args += 1
        elif opt in ("-c", "--config"):
            config = arg
            present_args += 1
        elif opt in ("-n", "--name"):
            name = arg

    assert present_args == 4, "help: run.py -d <base_dir> -p <wandb_project> -e <wandb_entity> -c <config> -n <run_name>"

    wandb.init(dir=base_dir, project=project, entity=entity, config=config, name=name)
    main(wandb.config)
