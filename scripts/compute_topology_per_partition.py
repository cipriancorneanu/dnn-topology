import os
import argparse
from config import NPROC, T, UPPER_DIM
import time

parser = argparse.ArgumentParser()
parser.add_argument('--save_path')
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--trial', default=0)
parser.add_argument('--part', default=0)
parser.add_argument('--epochs', nargs='+')
parser.add_argument('--thresholds', nargs='+', type=float)
parser.add_argument('--homology_type', default='static')
args = parser.parse_args()

path = args.save_path+"adjacency/"+args.net+"_"+args.dataset+"/"

for e in args.epochs:
    if args.homology_type == 'static':
        for t in args.thresholds:
            os.system("../cpp/symmetric_b1 "+path+"badj_epc"+str(e)+'_t{:1.4f}'.format(t)+"_trl"+args.trial+"_part"+args.part+".csv 1 0")
    elif args.homology_type == 'persistent':
        start = time.time()
        print("../dipha/full_to_sparse_distance_matrix "+str(T)+" "+path+"adj_epc{}_trl{}_part{}.bin ".format(e, args.trial, args.part)+path+"adj_epc{}_trl{}_part{}_{}.bin".format(e, args.trial, args.part, T))
        os.system("../dipha/full_to_sparse_distance_matrix "+str(T)+" "+path+"adj_epc{}_trl{}_part{}.bin ".format(e, args.trial, args.part)+path+"adj_epc{}_trl{}_part{}_{}.bin".format(e, args.trial, args.part, T))
        print(25*"-")
        print("Sparse distance matrix saved")
        print(25*"-")
        print("mpiexec -n "+str(NPROC)+" ../dipha/dipha --upper_dim "+str(UPPER_DIM)+" --benchmark  --dual "+path+"adj_epc{}_trl{}_part{}_{}.bin ".format(e, args.trial, args.part, T)+path+"adj_epc{}_trl{}_part{}_{}.bin.out".format( e, args.trial, args.part, T))
        os.system("mpiexec -n "+str(NPROC)+" ../dipha/dipha --upper_dim "+str(UPPER_DIM)+" --benchmark  --dual "+path+"adj_epc{}_trl{}_part{}_{}.bin ".format(e, args.trial, args.part, T)+path+"adj_epc{}_trl{}_part{}_{}.bin.out".format( e, args.trial, args.part, T))
        print("Topology computed in {} secs".format(time.time()-start))
