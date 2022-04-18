import argparse
import json
import random
import numpy as np
import mauve
import pickle
import glob
import os
import matplotlib.pyplot as plt
from nltk import tokenize
from nltk.corpus import stopwords
from utils import f1_score, rep_statistic


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="data/multi_outs/t5_xxl_wiki_t5_xl_gen_inbook_all.jsonl")
parser.add_argument('--eval_type', default="both")
parser.add_argument('--gram', default=1, type=int)
parser.add_argument('--rep_window', default=20, type=int)
parser.add_argument('--plot_divergence', action='store_true')
parser.add_argument('--eval_mauve', action='store_true')
args = parser.parse_args()

files = glob.glob(args.dataset)

if len(files) == 8:
    base_dir = os.path.dirname(files[0])
    assert all([os.path.dirname(x) == base_dir for x in files])
    files = ['pg19_gpt2_medium.jsonl', 'wiki_gpt2_medium.jsonl', 'pg19_gpt2_xl.jsonl', 'wiki_gpt2_xl.jsonl',
             'pg19_t5_xxl.jsonl', 'wiki_t5_xxl.jsonl', 'pg19_t5_xxl_descartes.jsonl', 'wiki_t5_xxl_descartes.jsonl']
    files = [os.path.join(base_dir, f) for f in files]

latex_token_overlap = []
latex_rep_score = []
random_latex_token_overlap = []
random_latex_rep_score = []
latex_gold_beats_gen = []

for file in files:

    with open(file, 'r') as f:
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
    gold_beats_gen = []

    for i in range(1):
        all_human = []
        all_gen = []
        num_tokens = []
        all_max_score = []
        for dd in data:
            all_human.append(dd['prefix'] + ' ' + dd['targets'][0])
            random_gen = random.choice(dd['targets'][1:])
            best_gen_idx = np.argmax(dd['scores'][1:]) + 1
            best_gen = dd['targets'][best_gen_idx]
            all_gen.append(dd['prefix'] + ' ' + random_gen)
            all_max_score.append(dd['prefix'] + ' ' + best_gen)
            num_tokens.append(len(best_gen.split()))

            token_overlaps["human"].append(
                f1_score(dd['targets'][0], dd['prefix'], stopwords=stopwords.words('english'), gram=args.gram)[0]
            )
            token_overlaps["random"].append(
                f1_score(random_gen, dd['prefix'], stopwords=stopwords.words('english'), gram=args.gram)[0]
            )
            token_overlaps["best"].append(
                f1_score(best_gen, dd['prefix'], stopwords=stopwords.words('english'), gram=args.gram)[0]
            )
            rep_scores["human"].append(rep_statistic(dd['prefix'], dd['targets'][0], window=args.rep_window))
            rep_scores["random"].append(rep_statistic(dd['prefix'], random_gen, window=args.rep_window))
            rep_scores["best"].append(rep_statistic(dd['prefix'], best_gen, window=args.rep_window))

            gold_beats_gen.extend([
                dd['scores'][0] > x for x in dd['scores'][1:]
            ])

        print(f"Results for {file}...")
        print(f"Human token overlap = {np.mean(token_overlaps['human']):.3f}")
        print(f"Random token overlap = {np.mean(token_overlaps['random']):.3f}")
        print(f"Best gen token overlap = {np.mean(token_overlaps['best']):.3f}")

        print(f"Human rep = {np.mean(rep_scores['human']):.3f}")
        print(f"Random rep = {np.mean(rep_scores['random']):.3f}")
        print(f"Best gen rep = {np.mean(rep_scores['best']):.3f}")

        print(f"Gold beats generation = {np.mean(gold_beats_gen)}")

        latex_token_overlap.append(np.mean(token_overlaps['best']))
        latex_rep_score.append(np.mean(rep_scores['best']))

        random_latex_token_overlap.append(np.mean(token_overlaps['random']))
        random_latex_rep_score.append(np.mean(rep_scores['random']))

        latex_gold_beats_gen.append(np.mean(gold_beats_gen))

        if i == 0 and args.eval_type == "both" and args.eval_mauve:
            mauve2 = mauve.compute_mauve(p_text=all_max_score, q_text=all_human, device_id=0, max_text_length=768, verbose=False)
            print(f"Max score mauve = {mauve2.mauve}")

        if args.eval_mauve:
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

        if args.eval_mauve:
            outputs = {
                "max_gen_mauve": mauve2,
                "random_gen_mauve": mauve1
            }
            with open(args.dataset + ".mauve.pkl", "wb") as f:
                pickle.dump(outputs, f)

print("Random gen latex --- ")
print(f"Latex token overlap = {' & '.join([f'{100 * x:.1f}' for x in random_latex_token_overlap])} & {100 * np.mean(random_latex_token_overlap):.1f}")
print(f"Latex rep = {' & '.join([f'{100 * x:.1f}' for x in random_latex_rep_score])} & {100 * np.mean(random_latex_rep_score):.1f}")


print("Best gen latex --- ")
print(f"Latex token overlap = {' & '.join([f'{100 * x:.1f}' for x in latex_token_overlap])} & {100 * np.mean(latex_token_overlap):.1f}")
print(f"Latex rep = {' & '.join([f'{100 * x:.1f}' for x in latex_rep_score])} & {100 * np.mean(latex_rep_score):.1f}")

print("Gold beats gen latex --- ")
print(f"Latex rep = {' & '.join([f'{100 * x:.1f}' for x in latex_gold_beats_gen])} & {100 * np.mean(latex_gold_beats_gen):.1f}")
