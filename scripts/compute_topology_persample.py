import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_path')
parser.add_argument('--net')
parser.add_argument('--dataset')
parser.add_argument('--trial', default=0)
parser.add_argument('--epochs', nargs='+')
parser.add_argument('--thresholds', nargs='+', type=float)
parser.add_argument('--sample', type=int)
args = parser.parse_args()


for e in args.epochs:
    for t in args.thresholds:
        os.system("../cpp/symmetric_b1 "+args.save_path+"adjacency/"+args.net+"_"+args.dataset+"/badj_epc"+str(e)+'_t{:1.2f}'.format(t)+"_trl"+args.trial+"_sample"+str(args.sample)+".csv 1 0")
