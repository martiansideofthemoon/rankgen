import argparse
import json
import random
import numpy as np
import mauve
import pickle
import matplotlib.pyplot as plt
from nltk import tokenize
from nltk.corpus import stopwords
from utils import f1_score


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="data/multi_outs/t5_xxl_wiki_t5_xl_gen_inbook_all.jsonl")
parser.add_argument('--eval_type', default="both")
parser.add_argument('--plot_divergence', action='store_true')
args = parser.parse_args()

with open(args.dataset, 'r') as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

for i in range(1):
    all_human = []
    all_gen = []
    num_tokens = []
    all_max_score = []
    for dd in data:
        all_human.append(dd['prefix'] + ' ' + dd['targets'][0])
        random_gen = random.choice(dd['targets'][1:])

        token_overlap_scores = [f1_score(x, dd['prefix'], stopwords=stopwords.words('english'))[0] for x in dd['targets'][1:]]
        best_token_overlap_idx = np.argmax(token_overlap_scores) + 1
        best_gen = dd['targets'][best_token_overlap_idx]
        all_gen.append(dd['prefix'] + ' ' + random_gen)
        all_max_score.append(dd['prefix'] + ' ' + best_gen)

        num_tokens.append(len(best_gen.split()))

    if i == 0 and args.eval_type == "both":
        mauve2 = mauve.compute_mauve(p_text=all_max_score, q_text=all_human, device_id=0, max_text_length=768, verbose=False)
        print(f"Max score mauve = {mauve2.mauve}")

    mauve1 = mauve.compute_mauve(p_text=all_gen, q_text=all_human, device_id=0, max_text_length=768, verbose=False)
    print(f"Random gen mauve = {mauve1.mauve}")

    if i == 0 and args.plot_divergence:
        plt.rcParams.update({'font.size': 16})
        plt.axis([0.0, 1.0, 0.0, 1.0])
        plt.plot(mauve1.divergence_curve[:, 0], mauve1.divergence_curve[:, 1], label="nucleus")
        plt.plot(mauve2.divergence_curve[:, 0], mauve2.divergence_curve[:, 1], label="rankgen")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                   fancybox=True, shadow=False, ncol=1)
        plt.ylabel("similarity to P")
        plt.xlabel("similarity to Q")
        plt.savefig('plot.pdf', bbox_inches="tight")

    outputs = {
        "token_overlap_mauve": mauve2,
        "random_gen_mauve": mauve1
    }
    with open(args.dataset + "token.mauve.pkl", "wb") as f:
        pickle.dump(outputs, f)
