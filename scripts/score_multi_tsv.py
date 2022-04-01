import argparse
import json
import random
import numpy as np
import mauve

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="data/multi_outs/t5_xxl_descartes_wiki_greedy.tsv")
args = parser.parse_args()

with open(args.dataset, 'r') as f:
    data = [x.split('\t') for x in f.read().strip().split("\n")]

all_human = []
all_gen = []
for i in range(0, len(data), 2):
    assert data[i][0] == data[i + 1][0]
    all_human.append(data[i][0] + ' ' + data[i][1])
    all_gen.append(data[i + 1][0] + ' ' + data[i + 1][1])

mauve1 = mauve.compute_mauve(p_text=all_gen, q_text=all_human, device_id=0, max_text_length=768, verbose=False)
print(mauve1.mauve)
