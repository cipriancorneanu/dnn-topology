__author__ = 'cipriancorneanu'
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import argparse
from sklearn.decomposition import PCA
import h5py
import os
import pickle as pkl
import scipy.stats
from sklearn import preprocessing 
import time
import pymetis
import itertools    

def correlation(x, y):
    return np.corrcoef(x,y)[0,1]


def kl(x, y):
    ''' 
    Return Kullback-Leibler divergence btw two probability density functions
    x, y: 1D nd array, sum(x)=1, sum(y)=1
    '''
    
    x[x==0]=0.00001
    y[y==0]=0.00001
    return scipy.stats.entropy(x, y)


def js(x, y):
    '''
    Return Jensen-Shannon divegence btw two probability density functions
    x, y: 1D ndarray
    '''

    return 0.5*kl(x,y) + 0.5*kl(y,x)


def corrpdf(signals):
    '''
    Compute pdf of correlations between signals
    signals: 2D ndarray, each row is a signal 
    '''

    ''' Get correlation matrix '''
    x = np.abs(np.nan_to_num(np.corrcoef(signals)))
    
    ''' Get upper triangular part (without diagonal) and vectorize'''
    x = x[np.triu_indices(x.shape[0], 1)]
    
    ''' Compute pdf'''
    pdf, _ = np.histogram(x, density=True)
    
    return pdf


def adjacency_correlation(signals):
    ''' Faster version of adjacency matrix with correlation metric '''
    signals = np.reshape(signals, (signals.shape[0], -1))
    return np.abs(np.nan_to_num(np.corrcoef(signals)))


def adjacency_l2(signal):
    ''' In this case signal is an 1XN array not a time series. 
    Builds adjacency based on L2 norm between node activations.
    '''
    x = np.tile(signal, (signal.size, 1))
    return np.sqrt((signal - signal.transpose())**2)


def binarize(M, binarize_t):
    ''' Binarize matrix. Real subunitary values. '''
    M[M>binarize_t] = 1
    M[M<=binarize_t] = 0
    
    return M
    

def adjacency(signals, metric=None):
    '''
    Build matrix A  of dimensions nxn where a_{ij} = metric(a_i, a_j).
    signals: nxm matrix where each row (signal[k], k=range(n)) is a signal. 
    metric: a function f(.,.) that takes two 1D ndarrays and outputs a single real number (e.g correlation, KL divergence etc).
    '''
    
    ''' Get input dimensions '''
    signals = np.reshape(signals, (signals.shape[0], -1))

    ''' If no metric provided fast-compute correlation  '''
    if not metric:
        return np.abs(np.nan_to_num(np.corrcoef(signals)))
        
    n, m = signals.shape
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            A[i,j] = metric(signals[i], np.transpose(signals[j]))

    ''' Normalize '''
    A = robust_scaler(A)
            
    return np.abs(np.nan_to_num(A))


def minmax_scaler(A):
    A = (A - A.min())/A.max()
    return A


def standard_scaler(A):
    return  np.abs((A - np.mean(A))/np.std(A))


def robust_scaler(A, quantiles=[0.05, 0.95]):
    a = np.quantile(A, quantiles[0])
    b = np.quantile(A, quantiles[1])
    return (A-a)/(b-a)


def adj2list(adj):
    return [[i for i,x in enumerate(row) if x == 1] for row in adj]


def signal_partition(signals, n_part=100, binarize_t=.5):
    signals = signal_concat(signals)
    print('Inside signal_partition signal concat shape is {}'.format(signals.shape))
    
    adj = adjacency_correlation(signals)
    badj = binarize(np.copy(adj), binarize_t)

    start = time.time()
    partition = pymetis.part_graph(n_part, adj2list(badj))[1]
    print('PyMetis Partition finished! in {} secs'.format(time.time()-start))

    node_splits = [[i for i, p in enumerate(partition) if p == val] for val in range(n_part)]
    splits = [[signals[indices, :] for indices in node_splits]]
   
    return node_splits, splits


def signal_splitting(signals, sz_chunk):
    splits = []
    
    for s in signals:
        s = np.reshape(s, (s.shape[0], np.prod(s.shape[1:])))
        sz = np.prod(np.shape(s)[1:])
        
        if sz > sz_chunk:
            splits.append([np.transpose(x) for x in np.array_split(s, sz/sz_chunk, axis=1)])
        else:
            splits.append([np.transpose(s)])
        
    return splits


def signal_dimension_adjusting(signals, sz_chunk):
    splits = []
    for s in signals:
        s = np.reshape(s, (s.shape[0], np.prod(s.shape[1:])))
        sz = np.prod(np.shape(s)[1:])
        
        if sz >= sz_chunk:
            [splits.append(np.transpose(x)) for x in np.array_split(s, sz/sz_chunk, axis=1)]
        else:
            splits.append([np.transpose(s)])
    for s in splits:
        print("splits size = ",len(s),len(s[0]))
    sp = [np.concatenate(list(zip(*splits))[i]) for i in range(len(splits[0]))]
    return sp


def signal_concat(signals):
    return np.concatenate([np.transpose(x.reshape(x.shape[0], -1)) for x in signals], axis=0)


def adjacency_set_correlation(splits):            
    set_averages = np.asarray([np.mean(x, axis=0) for x in splits])
    A = adjacency_correlation(set_averages)
    
    return A

def adjacency_correlation_distribution(splits, metric):            
    ''' Get correlation distribution for each split and build adjacency matrix between
    set of chunks using metric between distributions. '''

    ''' Compute correlation pdfs per split '''
    corrpdfs = [corrpdf(x) for layer in splits for x in layer]
    
    ''' Compute adjacency matrix'''
    A = adjacency(np.asarray(corrpdfs), metric=metric)
    
    return robust_scaler(A)


def build_density_adjacency(adj, density_t):
    ''' Binarize matrix '''
    total_edges = np.prod(adj.shape)
    t, t_decr = 1, 0.001
    while True:
        ''' Decrease threshold until density is met '''
        edges = np.sum(adj > t)
        density = edges/total_edges
        '''print('Threhold: {}; Density:{}'.format(t, density))'''
        
        if density > density_t:
            break

        t = t-t_decr
        
    return t


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

def get_discriminative_nodes(X, Y, ratio):
    X = X.astype(np.float64)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.001)
    
    print(np.max(X), np.min(X), np.mean(X))
    print(np.sum(X))

    clf.fit(X, Y)
    print('LDA score is {}'.format(clf.score(X, Y)))    

    return np.argsort(np.abs(clf.coef_))[:, -int(ratio*clf.coef_.size):]        


def get_index(shape, position):
    ''' '''
    return int(position[0]*shape[2]*shape[3] + position[1]*shape[3] + position[2])


def structure_conv2d(in_layer, out_layer, weights, kernel_size, stride, padding):
    ''' 
    Build adjacency matrix for a conv2d operation between two tensors.
    Comment: in this form padding and dilation are ignored and 
    kernel_size and stride are cosidered square that is stride[0]=stride[1]
    '''

    a = np.zeros((np.prod(in_layer.shape), np.prod(out_layer.shape)))
    '''
    print("structure built has shape {}".format(a.shape))
    print("weights have shape {}".format(weights.shape))
    '''
    n_in_channels, in_width, in_height = in_layer.shape[1:]
    n_out_channels, out_width, out_height = out_layer.shape[1:]
    
    ''' Input sliding window center coordinates '''
    i_star_list = [int(i) for i in np.arange((kernel_size[0]-1)/2, in_width-(kernel_size[0]-1)/2, stride[0])]
    j_star_list = [int(j) for j in np.arange((kernel_size[0]-1)/2, in_width-(kernel_size[0]-1)/2, stride[0])]

    ''' Output sliding window targets '''
    r_star_list = int((kernel_size[0]-1)/2) + np.asarray([i for i,_ in enumerate(i_star_list)], dtype=int)
    s_star_list = int((kernel_size[0]-1)/2) + np.asarray([i for i,_ in enumerate(j_star_list)], dtype=int)

    ''' Input space '''
    in_coords = [[x for x in itertools.product(np.arange(n_in_channels),np.arange(i_star-(kernel_size[0]-1)/2, i_star+(kernel_size[1]+1)/2),
                                               np.arange(j_star-(kernel_size[1]-1)/2, j_star+(kernel_size[1]+1)/2))] for i_star,j_star in itertools.product(i_star_list, j_star_list)]

    ''' Output space '''
    out_coords = [[x for x in itertools.product([r_star], [s_star], np.arange(n_out_channels))] for r_star, s_star in itertools.product(r_star_list, s_star_list)]

    ''' Iterate though input output pairs and fill structure '''
    weights = weights.data.cpu().numpy()
    for i, (inc,otc) in enumerate(zip(in_coords, out_coords)):
        '''
        print('Filling pair {}/{}'.format(i,len(in_coords)))
        print('{}:{} > {}'.format(i,inc,otc))
        print('{}> {}'.format(len(inc), len(otc)))
        print('-'*32)
        '''
        for i in range(n_out_channels):
            linear_x = [get_index(in_layer.shape, x) for x in inc]
            linear_y = get_index(out_layer.shape, otc[i])  
            '''print('linear_x: {} / linear_y: {}'.format(linear_x, linear_y))'''
            for j in range(n_in_channels):
                for ix,x in enumerate(linear_x[j*np.prod(kernel_size):(j+1)*np.prod(kernel_size)]):
                    q,r = divmod(ix,kernel_size[0])
                    '''print('a[{},{}] = weights[{},{},{},{}]'.format(x,linear_y,i,j,q,r))'''
                    a[x,linear_y] = weights[i,j,q,r]
    return a


def structure_maxpool2d(in_layer, out_layer, kernel_size, stride, padding, dilation):
    ''' 
    Build adjacency matrix for a maxpool2d operation between two tensors.
    Very similar to conv2d. They could be a slingle more abstract function.
    Comment: in this form padding and dilation are ignored and 
    kernel_size and stride are cosidered square that is stride[0]=stride[1]
    '''

    a = np.zeros((np.prod(in_layer.shape), np.prod(out_layer.shape)))
    n_in_channels, in_width, in_height = in_layer.shape[1:]
    n_out_channels, out_width, out_height = out_layer.shape[1:]
    
    ''' Input sliding window center coordinates '''
    i_star_list = j_star_list = [int(i) for i in np.arange((kernel_size-1)/2, in_width-(kernel_size-1)/2, stride)]

    ''' Output sliding window targets '''
    r_star_list = s_star_list = int(kernel_size/2) + np.asarray([i for i,_ in enumerate(i_star_list)], dtype=int)

    '''print('i_star_list {}\n j_star_list {}\n r_star_list {}\n s_star_list {}\n'.format(i_star_list, j_star_list, r_star_list, s_star_list))'''
    
    ''' Input space '''
    in_coords = [[x for x in itertools.product(np.arange(n_in_channels),np.arange(i_star-(kernel_size-1)/2, i_star+(kernel_size+1)/2),
                                               np.arange(j_star-(kernel_size-1)/2, j_star+(kernel_size+1)/2))] for i_star,j_star in itertools.product(i_star_list, j_star_list)]

    ''' Output space '''
    out_coords = [[x for x in itertools.product([r_star], [s_star], np.arange(n_out_channels))] for r_star, s_star in itertools.product(r_star_list, s_star_list)]

    ''' Iterate though input output pairs and fill structure '''
    for i, (inc,otc) in enumerate(zip(in_coords, out_coords)):
        '''
        print(i)
        print('{}:{} > {}'.format(i,inc,otc))
        print('{}> {}'.format(len(inc), len(otc)))
        print('-'*32)
        '''
        for i in range(n_out_channels):
            linear_x = [get_index(in_layer.shape, x) for x in inc]
            linear_y = get_index(out_layer.shape, otc[i])  
            '''print('linear_x: {} / linear_y: {}'.format(linear_x, linear_y))'''
            for j in range(n_in_channels):
                for ix,x in enumerate(linear_x[j*np.prod(kernel_size):(j+1)*np.prod(kernel_size)]):
                    q,r = divmod(ix,kernel_size)
                    '''print('a[{},{}] = weights[{},{},{},{}]'.format(x,linear_y,i,j,q,r))'''
                    a[x,linear_y] = 1

    return a


def structure_linear(weights):
    ''' Adjacency matrix for a linear operation (fully connected layer)'''
    return np.transpose(weights)


def get_node_number(model, x):
    ''' Get total number of nodes of model for a specific input '''
    return np.sum([np.prod(np.asarray(x.shape)) for x in model.forward_all_features(x)])

def get_features(model, x, view):
    print(x.shape)
    feats = model.forward_all_features(x)
    print(len(feats))
    return [feats[i] for i in view]
    
def get_operations(model, view):
    layers = list(model.features.children())
    layers.append(model.classifier)
    return [layers[i] for i in view]

def get_view(model, x):
    VIEW_VGG16 = {'ops': [24, 27, 30, 33],
                  'feats': [23, 26, 29, 32, 33]}
    tensors = get_features(model, x, VIEW_VGG16['feats'])
    layers = get_operations(model, VIEW_VGG16['ops'])
    
    return layers, tensors


def structure_from_view(model, x):
    '''
    Given a model (neural network), and an input x <tensor>
    return adjacency matrix that defines a graph correspoding
    to the structure of the model.
    '''
    layers, tensors = get_view(model, x)
    n_nodes = [np.prod(x.shape) for x in tensors]
    print('There are {} nodes in the considered view of the model'.format(np.sum(n_nodes)))
    A = np.zeros((np.sum(n_nodes), np.sum(n_nodes)))
        
    for i, (x, y, layer) in enumerate(zip(tensors[:-1], tensors[1:], layers)):
        layer_type = type(layer).__name__
        print('Layer {}: {} -> {} -> {}'.format(i, x.shape, layer_type, y.shape))
        
        if layer_type == 'Conv2d':
            weights, kernel_size, stride, padding = list(layer.parameters())[0], layer.kernel_size, layer.stride, layer.padding
            a = structure_conv2d(x, y, weights, kernel_size, stride, padding)
        elif layer_type == 'MaxPool2d':
            kernel_size, stride, padding, dilation = layer.kernel_size, layer.stride, layer.padding, layer.dilation 
            a = structure_maxpool2d(x, y, kernel_size, stride, padding, dilation)
        elif layer_type == 'Linear':
            weights = list(layer.parameters())[0]
            a = structure_linear(weights)
        else:
            pass
        
        ''' Fill structure '''        
        acc_n_nodes = [int(np.sum(n_nodes[:i])) for i in range(len(n_nodes)+1)]
        '''print(acc_n_nodes)'''
        start_in, end_in, start_out, end_out = acc_n_nodes[i], acc_n_nodes[i+1], acc_n_nodes[i+1], acc_n_nodes[i+2]
        '''print(start_in, end_in, start_out, end_out)'''
        print('A.shape={}, a.shape={}'.format(A.shape, a.shape))
        A[start_in:end_in, start_out:end_out] = a
        
    return A

