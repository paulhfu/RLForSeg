import os
import sys
import yaml

def add_dict(src, tgt):
    if not isinstance(src, dict):
        return src
    for key, val in src.items():
        tgt[key] = add_dict(val, AttrDict())
    return tgt

def dict_to_attrdict(src_dict):
    tgt = AttrDict()
    add_dict(src_dict, tgt)
    return tgt

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class YamlConf():

    def __init__(self, confir):
        conf_dir = os.path.join(os.getcwd(), confir)
        self.cfg = AttrDict()
        for arg in sys.argv[1:]:
            dirname = arg[:arg.find('=')]
            fname = arg[arg.find('=') + 1:]
            with open(os.path.join(conf_dir, dirname, fname + ".yaml")) as info:
                self.cfg[dirname] = add_dict(yaml.full_load(info), AttrDict())


