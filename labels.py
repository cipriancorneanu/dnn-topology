import numpy as np


''' Predefined permutation (size=128) '''
P128 = [30,  88,  93,  10,  53,  28,  22,   9, 116,  20,  51, 106,  14,
        76, 108, 119,  67,  82,  27,  39, 110,  68,  91,  74,  86, 117,
        56,  99,  66,  49,  11,  61,  65,   7,  58,  31,  35,  47,  23,
        96,  77,  25, 109,  12,  71, 123,  95,  48,  17,  44,   2, 101,
        118,   5,  59,  32, 122,  83,  78,  55,  54, 121,   8, 125,  97,
        57, 105, 120,  26,  43,  72,  40,  19, 115,  94,  89,  81,  64,
        70,  87,  29,  42,  46,  60,  37, 113,  41,   0,  92, 100,  24,
        75,  52,  90, 103,  84,   1,  80,  21, 111,   6,   3,   4,  79,
        50, 102, 112, 104,  45,  18,  62,  33, 127,  16,  38,  63,  85,
        124,  98,  36, 107,  73,  69,  34,  15, 114, 126,  13]


def accuracy(permuted):
    ordered = np.arange(len(permuted))
    correct = (permuted == ordered).sum()
    return 100.*correct/len(permuted)


def permutation(length, ratio=None):
    ordered = np.arange(length)
    permuted = np.random.permutation(length)

    if ratio:
        ratio = int(ratio*length)
        return np.concatenate((permuted[:ratio], ordered[ratio:]))
    else:
        return permuted

def identity(labels):
    ''' Do nothing on input '''
    return labels


class LPermutor():
    def __init__(self, perm, prop):
        self.perm = perm
        self.proportion = prop

    def run(self, labels):
        ''' Permute input using permutation matrix self.perm and proportion '''
        a = int(self.proportion*list(labels.size())[0])
        labels[:a] = labels[self.perm[:a]]
        return labels
        
        
class LBinarizer():
    def __init__(self, pivot):
        self.pivot = pivot

    def run(self, labels):
        ''' Binarize input. Put pivot to one, rest to zero.'''
        labels[labels != self.pivot] = 0
        labels[labels == self.pivot] = 1
        return labels

    
def load_manipulator(permute_labels=None, binarize_labels=-1): 
    ''' Define label manipulator '''
    if permute_labels:
        manipulator = LPermutor(P128, permute_labels).run
    elif binarize_labels >= 0:
        manipulator = LBinarizer(binarize_labels).run
    else:
        manipulator = identity

    return manipulator
 
    
if __name__ == '__main__':
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for p in x:
        x_ = np.copy(x)
        manipulator = LBinarizer(p).run
        print(p)
        print(x_)
        print(manipulator(x_))
    
