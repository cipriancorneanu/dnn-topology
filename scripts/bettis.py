import argparse
import pickle
import numpy as np 

def betti_integral(bettis, t_begin, t_end, delta=0.05):
    ''' Compute integral '''
    return np.sum([bettis['t_{:1.2f}'.format(x)] for x in np.arange(t_begin, t_end, delta)], axis=0)
    

def int_valerr(x):
    ''' int(x) with value error catch '''
    try:
        return int(x)
    except ValueError:
        return 0

    
def compile_bettis(fname, n_bettis=3):
    """ Read bettis from correspoding file """        
    with open(fname) as f:
        content = f.readlines()
    content = content[0].split(',')[1:1+n_bettis]
    
    return [int_valerr(x) for x in content]
