from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import os
import argparse
from utils import progress_bar
import numpy as np
import h5py
import pickle 
from utils import *
from models.utils import get_model, get_criterion
from passers import Passer
from savers import save_activations, save_checkpoint, save_losses
from loaders import *
from graph import *
from labels import load_manipulator

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--save_path')
parser.add_argument('--trial', default=0, type=int)
parser.add_argument('--epochs', nargs='+', type=int)
parser.add_argument('--split', default=0, type=int)
parser.add_argument('--kl', default=0, type=int)
parser.add_argument('--input_size', default=32, type=int)
parser.add_argument('--thresholds', nargs='+', type=float)
parser.add_argument('--filtration', default='nominal')
parser.add_argument('--permute_labels', default=0, type=float)
parser.add_argument('--binarize_labels', default=-1, type=int)

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
''' Meta-name to be used as prefix on all savings'''
oname = args.net + '_' + args.dataset + '/'
SAVE_DIR = args.save_path + 'adjacency/' + oname
START_LAYER = 3 if args.net in ['vgg', 'resnet'] else 0 
THRESHOLDS = args.thresholds

''' If save directory doesn't exist create '''
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)    

# Build models
print('==> Building model..')
net = get_model(args.net, args.dataset)
net = net.to(device)
    
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

''' Prepare criterion '''
criterion = nn.CrossEntropyLoss()

''' Define label manipulator '''
manipulator = load_manipulator(args.permute_labels, args.binarize_labels)
    
for epoch in args.epochs:
    print('==> Loading checkpoint for epoch {}...'.format(epoch))
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+ args.net + '_' + args.dataset + '/ckpt_trial_' + str(args.trial) + '_epoch_' + str(epoch)+'.t7')
    net.load_state_dict(checkpoint['net'])
    
    ''' Define passer and get activations '''
    functloader = loader(args.dataset+'_test', batch_size=100, subset=list(range(0, 1000)))
    passer = Passer(net, functloader, criterion, device)
    passer_test = Passer(net, functloader, criterion, device)
    passer_test.run(manipulator=manipulator)
    activs = passer.get_function()
    activs = signal_concat(activs)
    adj = adjacency(activs)
    print('The dimension of the adjacency matrix is {}'.format(adj.shape))
    print('Adj mean {}, min {}, max {}'.format(np.mean(adj), np.min(adj), np.max(adj)))

    ''' Write adjacency to binary. To use as DIPHA input for persistence homology '''
    save_dipha(SAVE_DIR + 'adj_epc{}_trl{}.bin'.format(epoch, args.trial), 1-adj)

    ''' Compute thresholds. If nominal, use args.thresholds directly, if density, compute nominal correspoding to edge densitites first. For static homology. '''
    if args.filtration == 'density':
        edge_t = [build_density_adjacency(adj, t) for t in args.thresholds]
        print('The edge thresholds correspoding to required densities are: {}'.format(edge_t))
    
        for et, dt in zip(edge_t, args.thresholds):
            badj = binarize(np.copy(adj), et)
            print('Taking T={}, density={}'.format(et, np.sum(badj)/np.prod(adj.shape)))
            np.savetxt(SAVE_DIR + 'badj_epc{}_t{:1.4f}_trl{}.csv'.format(epoch, dt, args.trial), badj, fmt='%d', delimiter=",")
    elif args.filtration == 'nominal':
        print('Size of adjacency matrix is {}'.format(adj.shape))
        for threshold in args.thresholds:
            badj = binarize(np.copy(adj), threshold)
            np.savetxt(SAVE_DIR + 'badj_epc{}_t{:1.4f}_trl{}.csv'.format(epoch, threshold, args.trial), badj, fmt='%d', delimiter=",")
