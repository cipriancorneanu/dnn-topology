import numpy as np

# For what threholds I am going to create binary graphs for computing topology
THRESHOLDS = np.arange(0.50, 1, 0.05)

# Where to save topology results
SAVE_PATH='./results_dnn_topology'

# DIPHA
DIPHA_MAGIC_NUMBER = 8067171840
ID = 7
T = 0.4

# Number of cores MPI can use
NPROC = 4

# Sets dimensions to compute persistent homology. If '2' only the first betti curve will be computed.
# For first and second betti curves set to '3' etc.
UPPER_DIM = 2
