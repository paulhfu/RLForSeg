import os
import sys
import yaml

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
                self.cfg[dirname] = self.add_dict(AttrDict(), yaml.full_load(info))

    def add_dict(self, src, tgt):
        if not isinstance(tgt, dict):
            return tgt
        for key, val in tgt.items():
            src[key] = self.add_dict(AttrDict(), val)
        return src