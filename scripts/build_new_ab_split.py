import argparse
import collections
import json
import csv
import os
import glob
import numpy as np
import pickle
import random
import tqdm
from datetime import datetime
from collections import Counter, defaultdict
from scipy.stats import kendalltau
from statsmodels.stats.inter_rater import fleiss_kappa


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="ab_tests/gpt2_medium_nucleus_vs_beam/Batch_355182_scarecrow.csv")
parser.add_argument('--leftover', default="A1HM7F2N458OAM")
parser.add_argument('--replacement', default="AOZAWM1GLMJPC")
args = parser.parse_args()

data = []
header = None
files = args.dataset.split(",")
for fl in files:
    curr_data = []
    with open(fl, "r") as f:
        spamreader = csv.reader(f)
        for row in spamreader:
            curr_data.append(row)
    header = curr_data[0]
    data.extend(curr_data[1:])

worker_ids = list(set([x[15] for x in data]))
worker_ids.sort()
all_hit_ids = list(set([x[0] for x in data]))
hit_dict = {x[0]: x for x in data}

hits_worker_leftover = {x[0]: 1 for x in data if x[15] == args.leftover}
hits_replacement = {x[0]: 1 for x in data if x[15] == args.replacement}

other_hits = [x for x in all_hit_ids if x not in hits_worker_leftover and x not in hits_replacement]

output = [["Prefix", "First", "Second", "Order", "Dataset", "Model", "Old HIT ID"]]

for x in other_hits:
    orig_hit = hit_dict[x]
    output.append(orig_hit[27:33] + [x])

with open(os.path.dirname(args.dataset) + "/leftover.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(output)
