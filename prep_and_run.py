from rag_utils import find_dense_subgraphs
import numpy as np
edges = np.array([[0, 3], [1, 2], [2, 1], [2, 3], [3, 1]])
find_dense_subgraphs([edges], [2])


import subprocess
from cluster.easybuild_replace import replace_torch
import site
import os

# env_location = site.getsitepackages()[0]
# replace_torch()
# # os.environ['PYTHONPATH'] += ':' + env_location
# print("found env site: " + env_location)
# python_executable = os.path.join("/", *os.path.normpath(env_location).split(os.sep)[:-3], "bin/python")
# subprocess.run(["alias", f"python={python_executable}"], stdout=subprocess.PIPE)
# print("###################")
# print(subprocess.run(["which", "python"], stdout=subprocess.PIPE))

if __name__ == '__main__':
    import matplotlib

    matplotlib.use('Agg')
    import torch
    from run import main
    a = torch.tensor([1,2,3,4,5]).bool() | torch.tensor([1,2,3,4,5]).bool()
    print(a)
    # mp.set_start_method('spawn', force=True)
    main()
