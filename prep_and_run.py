import matplotlib
matplotlib.use('Agg')
from torch import multiprocessing as mp
from run import main

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
