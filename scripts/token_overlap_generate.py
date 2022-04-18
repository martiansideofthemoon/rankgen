import argparse
import json
import random
import numpy as np
import mauve
import pickle
import matplotlib.pyplot as plt
from nltk import tokenize
from nltk.corpus import stopwords
from utils import f1_score, rep_statistic
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="data/multi_outs/t5_xxl_wiki_t5_xl_gen_inbook_all.jsonl")
parser.add_argument('--output_file', default="data/multi_outs/t5_xxl_wiki_t5_xl_gen_inbook_all.jsonl")
parser.add_argument('--eval_type', default="both")
parser.add_argument('--rep_window', default=20, type=int)
parser.add_argument('--plot_divergence', action='store_true')
parser.add_argument('--eval_mauve', action='store_true')
args = parser.parse_args()

with open(args.dataset, 'r') as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

token_overlaps = {
    "human": [],
    "random": [],
    "best": []
}
rep_scores = {
    "human": [],
    "random": [],
    "best": []
}

for i in range(1):
    all_human = []
    all_gen = []
    num_tokens = []
    all_max_score = []
    for dd in tqdm.tqdm(data):
        all_human.append(dd['prefix'] + ' ' + dd['targets'][0])
        random_gen = random.choice(dd['targets'][1:])

        token_overlap_scores = [f1_score(x, dd['prefix'], stopwords=stopwords.words('english'))[0] for x in dd['targets']]

        dd['scores'] = token_overlap_scores

    output_txt = "\n".join([json.dumps(x) for x in data]) + "\n"
    with open(args.output_file, 'w') as f:
        f.write(output_txt)
