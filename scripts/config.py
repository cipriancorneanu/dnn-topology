import numpy as np

# For what threholds I am going to create binary graphs for computing topology
THRESHOLDS = np.arange(0.50, 1, 0.05)

# Where to save topology results
SAVE_PATH='/data/data1/datasets/cvpr2019/'

''' DIPHA '''
DIPHA_MAGIC_NUMBER = 8067171840
ID = 7
NPROC = 10
T = 0.4
UPPER_DIM = 2
