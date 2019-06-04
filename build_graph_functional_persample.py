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
from utils import *
from models.utils import get_model, get_criterion
from passers import Passer, get_accuracy
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
parser.add_argument('--n_samples', default=10, type=int)
parser.add_argument('--input_size', default=32, type=int)
parser.add_argument('--thresholds', nargs='+', type=float)
parser.add_argument('--permute_labels', default=0, type=float)
parser.add_argument('--binarize_labels', default=-1, type=int)
parser.add_argument('--select_nodes', default=0, type=int)
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
if args.dataset in ['cifar10', 'cifar10_gray', 'vgg_cifar10_adversarial', 'imagenet']:
    criterion = nn.CrossEntropyLoss()
elif args.dataset in ['mnist', 'mnist_adverarial']:
    criterion = F.nll_loss

''' Define label manipulator '''
manipulator = load_manipulator(args.permute_labels, args.binarize_labels)

''' Instead of building graph on the entire set of nodes, pick a subset '''
if args.select_nodes:
    ''' get activations '''
    subsettestloader = loader(args.dataset+'_train',  batch_size=100, sampling=args.binarize_labels)
    passer = Passer(net, subsettestloader, criterion, device)
    manipulator = load_manipulator(args.permute_labels, args.binarize_labels)

    activs = passer.get_function()
    activs = signal_concat(activs)

    ''' get correct and wrong predictions '''
    gts, preds = passer.get_predictions(manipulator=manipulator)    
    labels = [int(x) for x in gts==preds]

    ''' compute discriminative nodes '''
    nodes = np.concatenate(get_discriminative_nodes(np.transpose(activs), labels, 0.1))
else:
    nodes = np.arange(0, len(activs))

    
for epoch in args.epochs:
    print('==> Loading checkpoint for epoch {}...'.format(epoch))
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+ args.net + '_' + args.dataset + '/ckpt_trial_' + str(args.trial) + '_epoch_' + str(epoch)+'.t7')
    net.load_state_dict(checkpoint['net'])
    
    ''' Functional graph per sample'''
    dataloader = loader(args.dataset+'_test', batch_size=1, sampling=args.binarize_labels)
    passer = Passer(net, dataloader, criterion, device)
    preds, gts = [], []
    
    for sample, (inputs, targets) in enumerate(dataloader):
        if sample >= args.n_samples: break
        
        targets = manipulator(targets)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        accuracy = get_accuracy(outputs, targets)
        
        preds.append(outputs.cpu().data.numpy().argmax(1))
        gts.append(targets.cpu().data.numpy())
   
        print('Computing functional graph for sample {}/{} -- Acc: {:1.2f}'.format(sample, args.n_samples, accuracy))

        activs = signal_concat([f.cpu().data.numpy().astype(np.float16) for f in net.module.forward_features(inputs)])[nodes]
        adj = adjacency_l2(activs)
        adj = robust_scaler(adj, quantiles=[0.0, 1.0])

        for threshold in args.thresholds:
            badj = binarize(np.copy(adj), threshold)
            np.savetxt(SAVE_DIR + 'badj_epc{}_t{:1.2f}_trl{}_sample{}.csv'.format(epoch, threshold, args.trial, sample), badj, fmt='%d', delimiter=",")

    ''' Save labels '''
    lbls = [int(x) for x in np.concatenate(preds) == np.concatenate(gts)]
    np.save(SAVE_DIR+'lbls_epc{}_trl{}'.format(epoch, args.trial), lbls)
