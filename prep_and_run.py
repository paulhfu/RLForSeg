import matplotlib
matplotlib.use('Agg')
import torch.multiprocessing as mp
from run import main
from utils.yaml_conv_parser import YamlConf
import os
import sys
import yaml


if __name__ == '__main__':
    # os.nice(15)
    main(YamlConf("conf").cfg)
