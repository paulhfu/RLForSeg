import matplotlib
matplotlib.use('Agg')
import torch.multiprocessing as mp
from run import no_mp_main, main
from utils.yaml_conv_parser import YamlConf
import os
import sys
import yaml


if __name__ == '__main__':
    # os.nice(15)
    mp.set_start_method('spawn', force=True)
    main(YamlConf("conf").cfg)
