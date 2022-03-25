import argparse
import json
from utils import f1_score
import nltk
import tqdm
import numpy as np
from nltk import tokenize
from nltk.corpus import stopwords


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="data/t5_xl_all_domains_pg19_random.jsonl")
parser.add_argument('--num_negatives', default=10, type=int)
args = parser.parse_args()

avg_score = []
all_score = []

if args.dataset.endswith(".jsonl"):
    with open(args.dataset, "r") as f:
        data = [json.loads(x) for x in f.read().strip().split("\n")]
    for idx, dd in tqdm.tqdm(enumerate(data), total=len(data)):
        prefix = dd['prefix']
        candidates = [dd['suffix']] + dd['negatives']
        assert len(candidates) == args.num_negatives + 1
        overlap_scores = [f1_score(x, prefix, stopwords=stopwords.words('english'))[0] for x in candidates]
        avg_score.append(np.mean([overlap_scores[0] > y for y in overlap_scores[1:]]))
        all_score.append(all([overlap_scores[0] > y for y in overlap_scores[1:]]))

        if (idx + 1) % 10000 == 0:
            print(f"{np.mean(avg_score):.4f} average ({len(avg_score)} instances), {np.mean(all_score):.4f} all ({len(all_score)} instances)")

elif args.dataset.endswith(".tsv"):
    with open(args.dataset, "r") as f:
        data = [x.split("\t") for x in f.read().strip().split("\n")]
    for idx in tqdm.tqdm(range(0, len(data), args.num_negatives + 1)):
        prefix = data[idx][0]
        candidates = []
        for jdx in range(args.num_negatives + 1):
            assert data[idx + jdx][0] == prefix
            candidates.append(data[idx + jdx][1])
        assert len(candidates) == args.num_negatives + 1
        overlap_scores = [f1_score(x, prefix, stopwords=stopwords.words('english'))[0] for x in candidates]
        avg_score.append(np.mean([overlap_scores[0] > y for y in overlap_scores[1:]]))
        all_score.append(all([overlap_scores[0] > y for y in overlap_scores[1:]]))

        if (idx + 1) % 10000 == 0:
            print(f"{np.mean(avg_score):.4f} average ({len(avg_score)} instances), {np.mean(all_score):.4f} all ({len(all_score)} instances)")

print(f"{np.mean(avg_score):.4f} average ({len(avg_score)} instances), {np.mean(all_score):.4f} all ({len(all_score)} instances)")