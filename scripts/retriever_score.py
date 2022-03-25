import argparse
import json
from utils import f1_score
import nltk
import tqdm
import numpy as np
from nltk import tokenize
from nltk.corpus import stopwords


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="data/t5_xl_all_domains_pg19_hard.jsonl")
args = parser.parse_args()

with open(args.dataset, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

avg_score = []
all_score = []

for idx, dd in tqdm.tqdm(enumerate(data), total=len(data)):
    prefix = dd['prefix']
    candidates = [dd['suffix']] + dd['negatives']
    overlap_scores = [dd['suffix_score']] + dd['negative_scores']
    assert len(candidates) == 11
    avg_score.append(np.mean([overlap_scores[0] > y for y in overlap_scores[1:]]))
    all_score.append(all([overlap_scores[0] > y for y in overlap_scores[1:]]))

    if (idx + 1) % 10000 == 0:
        print(f"{np.mean(avg_score):.4f} average ({len(avg_score)} instances), {np.mean(all_score):.4f} all ({len(all_score)} instances)")

print(f"{np.mean(avg_score):.4f} average ({len(avg_score)} instances), {np.mean(all_score):.4f} all ({len(all_score)} instances)")
