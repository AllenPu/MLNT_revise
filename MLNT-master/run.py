import os
import argparse

parser = argparse.ArgumentParser(description='MLNT')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset: cifar10 / cifar100')
parser.add_argument('--noise_rate', default=0.4, type= float)
args = parser.parse_args()

base = "python ./baseline.py --dataset"
dataset = args.dataset
n_r = "--noise_rate"
noise = args.noise_rate
comd = str(base) + ' ' + str(dataset) + ' ' + str(n_r) + ' ' + str(noise)
base_2 = "python ./main.py --dataset"
comd_2 = str(base_2) + ' ' + str(dataset) + ' ' + str(n_r) + ' ' + str(noise)

os.system(comd)
os.system(comd_2)