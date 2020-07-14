import matplotlib.pyplot as plt
import scipy.ndimage
from bettis import *
import argparse
import os
from config import SAVE_PATH, MAX_EPSILON

parser = argparse.ArgumentParser()
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--trial', default=0)
parser.add_argument('--epochs', nargs='+', type=int)
parser.add_argument('--dim', default=1, type=int)
args = parser.parse_args()

directory = os.path.join(SAVE_PATH, args.net+'_'+args.dataset+'/')
trial = 0

for epc in args.epochs:
    #  read persistent diagram from persistent homology output
    birth, death = read_results(directory, epc, trl=args.trial, max_epsilon=MAX_EPSILON, dim=args.dim, persistence=0.02)

    if len(birth) > 0:
        #  compute betti curve from persistent diagram
        x, betti = pd2betti(birth, death)

        #  filter curve for improved visualization
        filter_size = int(len(betti) / 10)
        betti = scipy.ndimage.filters.uniform_filter1d(betti, size=filter_size, mode='constant')

        # plor curve
        plt.xlabel('$\epsilon$')
        plt.ylabel('#cavities')
        plt.plot(x, betti, label='epc_' + str(epc))
        plt.legend()
        plt.title('betti_{}'.format(args.dim))

        # compute life and midlife
        life = pd2life(birth, death)
        midlife = pd2midlife(birth, death)
        print('EPC = {}, LIFE = {}, MIDLIFE = {}'.format(epc, life, midlife))
    else:
        print('The persistence diagram is empty!')

plt.show()