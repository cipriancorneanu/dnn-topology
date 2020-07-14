__author__ = 'cipriancorneanu'

import numpy as np
import array


def read_bin(fname):
    header = array.array("L")
    values = array.array("d")

    with open(fname, mode='rb') as file:  # b is important -> binary
        header.fromfile(file, 3)
        values.fromfile(file, int(header[2] * header[2]))
        values = list(values)

    values = np.asarray([float("{0:.2f}".format(1 - x)) for x in np.asarray(values)])
    values = np.reshape(values, (header[2], header[2]))

    return values


def read_bin_out(fname):
    '''
    :param fname: Binary file name as written by DIPHA
    :return: contents of binary file name (dimensions, birth_values, death_values)
    '''
    header = array.array("L")
    dims = array.array("l")
    values = array.array("d")

    with open(fname, mode='rb') as file:  # b is important -> binary
        header.fromfile(file, 3)
        dims.fromfile(file, 3 * header[2])
        dims = list(dims[::3])
        file.seek(24)
        values.fromfile(file, 3 * (header[2]))
        values = list(values)
        birth_values = values[1::3]
        death_values = values[2::3]

    return dims, birth_values, death_values


def read_betti(fname, dimension, persistence):
    '''
    Read binary file name from DIPHA and transform to betti with normed x_axis
    '''
    dims, b_values, d_values = read_bin_out(fname)
    x, betti = pd2betti(b_values, d_values)
    return norm_x_axis(x, betti, np.linspace(0.05, 0.4, 200))


def read_pd(fname, dimension, persistence):
    dims, birth, death = read_bin_out(fname)

    d = [i for i, x in enumerate(dims) if x == dimension]

    birth = [float("{0:.4f}".format(x)) for x in np.asarray(birth)[d]]
    death = [float("{0:.4f}".format(x)) for x in np.asarray(death)[d]]

    ''' Filter out low persistence '''
    filter = [i for i, (xb, xd) in enumerate(zip(birth, death)) if (xd - xb) > persistence]
    birth = [birth[x] for x in filter]
    death = [death[x] for x in filter]

    return birth, death


def pd2betti(birth, death):
    x = sorted(birth + death)

    b_x = [x.index(item) for item in birth]
    b_y = [x.index(item) for item in death]

    delta_birth = np.zeros_like(x, dtype=np.int)
    delta_death = np.zeros_like(x, dtype=np.int)

    acc = 0
    for item in b_x:
        acc = acc + 1
        delta_birth[item:] = acc

    acc = 0
    for item in b_y:
        acc = acc + 1
        delta_death[item:] = acc

    return x, delta_birth - delta_death


def betti_max(x, curve):
    return np.max(curve)


def pd2betti_max(birth, death):
    x, betti = pd2betti(birth, death)
    return np.max(betti)


def epsilon_max(x, curve):
    return x[np.argmax(curve)]


def pd2epsilon_max(birth, death):
    x, betti = pd2betti(birth, death)
    return x[np.argmax(betti)]


def pd2life(birth, death):
    return np.mean(np.asarray(death) - np.asarray(birth))


def pd2midlife(birth, death):
    return np.mean(np.asarray(death) + np.asarray(birth) / 2)


def pd2mullife(birth, death):
    return np.mean(np.asarray(death) / np.asarray(birth))


def pd2amplitude(birth, life):
    return len(birth)


def compute_integral(curve):
    return np.sum(curve)


'''
def compute_auc(curve):
    #:param curve: list of values
    #:return: AUC for curve
    
    auc = [np.sum(curve[:end])/np.sum(curve) for end in range(len(curve))]
    return auc
'''

'''
def plot(axes, data, epochs, i_epochs, N):
    for i, (ax, i_epcs) in enumerate(zip(axes, i_epochs)):
        for i_epc in i_epcs:
            x, betti = data[i_epc]
            betti = np.array([betti[np.argmin([np.abs(a-b) for b in x])] for a in x_ref])
            ax.semilogx(x_ref, betti/N, label='epc{}'.format(epochs[i_epc]))
            ax.set_ylabel("cavs/node")
            ax.legend()
            ax.grid()
'''


def norm_x_axis(x, curve, x_ref):
    x_axis = np.array([curve[np.argmin([np.abs(a - b) for b in x])] for a in x_ref])
    # print("{}s".format(time.time()-start))
    return x_axis


def read_results_part(path, epcs, parts, trl, dim, persistence=0.04):
    return [[read_betti(path + 'adj_epc{}_trl{}_part{}_0.4.bin.out'.format(epc, trl, part), dimension=dim,
                        persistence=persistence) for epc in epcs] for part in parts]


def read_results(path, epc, trl, max_epsilon=0.4, dim=1, persistence=0.01):
    return read_pd(path + 'adj_epc{}_trl{}_{}.bin.out'.format(epc, trl, max_epsilon), dimension=dim, persistence=persistence)


'''
def evaluate_node_importance(adj, epsilon):
    node_importance = np.zeros(adj.shape[0])
    adj[adj<epsilon]=0
    adj[adj>=epsilon]=1
    importance = np.sum(adj, axis=0)
    print(np.sort(importance))
    return  np.argsort(importance)[::-1]


def compute_node_importance(net, dataset, epcs, trl):
    data = read_results(root+'results_15_04/lenet_mnist/', epcs_lenet, trl, dim=1)

    x_ref = np.linspace(0.05, 0.4, 200)
    maxcav= [np.argmax(norm_x_axis(epc[0], epc[1], x_ref)) for epc in part]

    # Get epc and eps
    epc, eps = 0 , 0

    # Evaluate node importance
    fname = root + 'adj_epc{}_trl{}.bin'.format(epc, trl)
    adj = read_bin_out(fname, dimension=1, persistence=0.03)
    node_importance = evaluate_node_importance(adj, epsilon=eps)

    return node_importance
'''


def get_data(root, nets, datasets, trials, epcs):
    n_nodes = {'mlp_300_100': 400, 'mlp_300_200_100': 600, 'conv_2': 650, 'conv_4': 906, 'alexnet': 1162, 'conv_6': 1418,
           'resnet18': 1736, 'vgg16': 1930, 'resnet34': 1736, 'resnet50': 6152}

    pts = []
    for i_net, net in enumerate(nets):
        for i_dataset, dataset in enumerate(datasets):
            directory = root + net + '_' + dataset + '/'
            if os.path.exists(directory):
                for i, epc in enumerate(epcs):
                    # If file exists, read and plot
                    for trial in trials:
                        if os.path.exists(directory + 'adj_epc' + str(epc) + '_trl{}_0.4.bin.out'.format(trial)):
                            # read data
                            data = read_results(directory, epc, trl=trial, dim=1, persistence=0.02)
                            if len(data) > 0:
                                x_ref = np.linspace(0.05, 0.4, 200)
                                # read generalization gap
                                with open(root + 'losses/' + net + '_' + dataset + '/stats_trial_' + str(trial) + '.pkl',
                                          'rb') as f:
                                    loss = pkl.load(f)

                                    if dataset in ['bp4d', 'disfa']:
                                        acc_tr, acc_te = loss[0]['acc_tr'][epc], loss[0]['acc_te'][epc]
                                        ggap = 100*(acc_tr-acc_te)
                                    else:
                                        acc_tr, acc_te = loss[epc]['acc_tr'], loss[epc]['acc_te']
                                        ggap  = (acc_tr-acc_te)

                                    print(net, dataset, trial, ggap)

                                    if ggap > 1 and ggap < 100:
                                        pts.append({'net':net, 'dataset':dataset, 'ggap':ggap, 'trial':trial, 'epc': epc, 'pd':data})
                        else:
                            # print("No such file {}".format(directory+'adj_epc'+str(epc)+'_trl'+str(trial)+'_0.4.bin.out'))
                            pass

    return pts


def extend(item, t_summary, key):
    item[key] = t_summary(item['pd'][0], item['pd'][1])
    return item


def betti_integral(bettis, t_begin, t_end, delta=0.05):
    """ Compute integral """
    return np.sum([bettis['t_{:1.2f}'.format(x)] for x in np.arange(t_begin, t_end, delta)], axis=0)


def int_valerr(x):
    """ int(x) with value error catch """
    try:
        return int(x)
    except ValueError:
        return 0


def compile_bettis(fname, n_bettis=3):
    """ Read bettis from correspoding file """
    with open(fname) as f:
        content = f.readlines()
    content = content[0].split(',')[1:1 + n_bettis]

    return [int_valerr(x) for x in content]
