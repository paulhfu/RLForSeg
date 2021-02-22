import os
import site
import subprocess
from shutil import rmtree


def get_easybuild_lib(easybuild_name, package_name):
    easybuild_script = os.path.abspath(os.path.join(os.path.split(__file__)[0], 'expand_easybuild_lib.sh'))
    easybuild_lib = subprocess.run([easybuild_script, easybuild_name, package_name], stdout=subprocess.PIPE)
    easybuild_lib =easybuild_lib.stdout.decode('utf-8')
    easybuild_lib = os.path.split(easybuild_lib)[0]
    return easybuild_lib

def easybuild_replace(package_name, easybuild_name):
    env_name = os.environ['CONDA_DEFAULT_ENV']
    env_lib = site.getsitepackages()
    assert len(env_lib) == 1
    env_lib = env_lib[0]
    conda_lib = os.path.join(env_lib, package_name)
    easybuild_lib = get_easybuild_lib(easybuild_name, package_name)
    os.environ["PYTHONEX"] = os.path.join("/", *os.path.normpath(env_lib).split(os.sep)[:-3], "bin/python")

    if os.path.exists(conda_lib):
        if os.path.islink(conda_lib):
            print("Found existing linked package", package_name, "at", conda_lib, "and removing it")
            os.unlink(conda_lib)
        elif os.path.isdir(conda_lib):
            print("Found existing", package_name, "at", conda_lib, "and removing it")
            rmtree(conda_lib)
        else:
            raise RuntimeError

    print("source: " + easybuild_lib)
    print("dest: " + conda_lib)
    os.symlink(easybuild_lib, conda_lib)
    return env_lib

def local_replace(package_name, path):
    env_name = os.environ['CONDA_DEFAULT_ENV']
    env_lib = site.getsitepackages()
    assert len(env_lib) == 1
    env_lib = env_lib[0]
    conda_lib = os.path.join(env_lib, package_name)

    if os.path.exists(conda_lib):
        if os.path.islink(conda_lib):
            print("Found existing linked package", package_name, "at", conda_lib, "and removing it")
            os.unlink(conda_lib)
        elif os.path.isdir(conda_lib):
            print("Found existing", package_name, "at", conda_lib, "and removing it")
            rmtree(conda_lib)
        else:
            raise RuntimeError

    print("source: " + path)
    print("dest: " + conda_lib)
    os.symlink(path, conda_lib)
    return env_lib


# TODO we don't have the easybuild package for this yet
def replace_torchvision():
    pass

if __name__=="__main__":
    local_replace('nifty', '/g/kreshuk/pape/Work/software/bld/py38/nifty/python/nifty')
    # local_replace('torch_scatter', '/g/kreshuk/hilt/projects/pytorch_scatter/python38cud11install/torch_scatter')
    # local_replace('rag_utils', '/g/kreshuk/hilt/projects/graph_utils/rag_utils/python/rag_utils')
    easybuild_replace('torch', 'PyTorch')