import argparse
import json
import csv
import os
import numpy as np
import random
from datetime import datetime
from scipy.stats import kendalltau


parser = argparse.ArgumentParser()
parser.add_argument('--input', default="ab_tests/gold_neg_wiki_random/input.csv")
parser.add_argument('--dataset', default="ab_tests/gold_neg_wiki_random/Batch_351346_batch_results.csv")
parser.add_argument('--num_instances', default=200)
args = parser.parse_args()

data = []
with open(args.dataset, "r") as f:
    spamreader = csv.reader(f)
    for row in spamreader:
        data.append(row)

data_input = []
with open(args.input, "r") as f:
    spamreader = csv.reader(f)
    for row in spamreader:
        data_input.append(row)

header = data[0]
data = [x for x in data[1:] if x[15] != 'A3PDXN2ET2Q9CO']

correct = 0
wrong = 0
verdict = []

for dd in data:
    if dd[-1] == "Text 1" and dd[-4] == "suffix,negative":
        correct += 1
        verdict.append(1)
    elif dd[-1] == "Text 2" and dd[-4] == "negative,suffix":
        correct += 1
        verdict.append(1)
    else:
        verdict.append(0)

time_taken = [int(dd[23]) for dd in data]
timestamps = [datetime.strptime(dd[18].replace('PDT ', ''), '%a %b %d %H:%M:%S %Y').timestamp() for dd in data]

print(correct / len(data))
print(len(data))
import pdb; pdb.set_trace()
pass
