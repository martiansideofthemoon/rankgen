import argparse
import csv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="data/story_cloze_spring_2016_val.tsv")
args = parser.parse_args()

data = []

with open(args.dataset, 'r') as f:
    data = [x.split("\t") for x in f.read().strip().split("\n")]

prefix_lens = []
suffix_lens = []

for dd in data:
    prefix_lens.append(len(dd[0].split()))
    suffix_lens.append(len(dd[1].split()))

print(f"Average prefix = {np.mean(prefix_lens)}")
print(f"Average suffix = {np.mean(suffix_lens)}")
