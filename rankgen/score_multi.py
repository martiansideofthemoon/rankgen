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
import spacy
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="data/multi_outs/t5_xxl_wiki_t5_xl_gen_inbook_all.jsonl")
parser.add_argument('--eval_type', default="max")
parser.add_argument('--gram', default=1, type=int)
parser.add_argument('--rep_window', default=20, type=int)
parser.add_argument('--plot_divergence', action='store_true')
parser.add_argument('--eval_mauve', action='store_true')
parser.add_argument('--eval_pos_overlap', action='store_true')
args = parser.parse_args()

files = glob.glob(args.dataset)

base_dir = os.path.dirname(files[0])
assert all([os.path.dirname(x) == base_dir for x in files])
files = ['pg19_t5_xxl_descartes.jsonl', 'wiki_t5_xxl_descartes.jsonl',
         'pg19_gpt2_medium.jsonl', 'wiki_gpt2_medium.jsonl',
         'pg19_gpt2_xl.jsonl', 'wiki_gpt2_xl.jsonl',
         'pg19_t5_xxl.jsonl', 'wiki_t5_xxl.jsonl']
files = [os.path.join(base_dir, f) for f in files]

if args.eval_pos_overlap:
    nlp = spacy.load("en_core_web_sm")
latex_token_overlap = []
latex_rep_score = []
random_latex_token_overlap = []
random_latex_rep_score = []
latex_gold_beats_gen = []
latex_mauve = []

random_latex_token_overlap_ents = []
latex_token_overlap_ents = []

for file in files:
    if not os.path.exists(file):
        continue
    with open(file, 'r') as f:
        data = [json.loads(x) for x in f.read().strip().split("\n")]

    data_dict = {x["prefix"]: x for x in data}
    if "wiki_" in file:
        with open("data/multi_outs/t5_xxl_descartes_wiki_ppl.jsonl", "r") as f:
            raw_inp_data = [json.loads(x) for x in f.read().strip().split("\n")]
        for rid in raw_inp_data:
            assert rid["prefix"] in data_dict
            assert rid["targets"][0] == data_dict[rid["prefix"]]["targets"][0]
    elif "pg19_" in file:
        with open("data_new/ppl/pg19_t5_xxl.jsonl", "r") as f:
            raw_inp_data = [json.loads(x) for x in f.read().strip().split("\n")]
        for rid in raw_inp_data:
            assert rid["prefix"] in data_dict
            assert rid["targets"][0] == data_dict[rid["prefix"]]["targets"][0]


    token_overlaps = {
        "human": [],
        "random": [],
        "best": []
    }
    if os.path.exists(file + ".ent_overlap.pkl"):
        with open(file + ".ent_overlap.pkl", "rb") as f:
            token_overlaps_ents = pickle.load(f)
    else:
        token_overlaps_ents = {
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
        num_tokens_random = []
        num_tokens = []
        all_max_score = []
        for dd in tqdm.tqdm(data):
            all_human.append(dd['prefix'] + ' ' + dd['targets'][0])
            random_gen = random.choice(dd['targets'][1:])
            best_gen_idx = np.argmax(dd['scores'][1:]) + 1
            best_gen = dd['targets'][best_gen_idx]
            all_gen.append(dd['prefix'] + ' ' + random_gen)
            all_max_score.append(dd['prefix'] + ' ' + best_gen)
            num_tokens.append(len(best_gen.split()))
            num_tokens_random.append(len(random_gen.split()))

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

            if args.eval_pos_overlap and not os.path.exists(file + ".ent_overlap.pkl"):
                prefix_nlp = nlp(dd['prefix'])
                best_nlp = nlp(best_gen)
                prefix_ents = " ".join([x.lemma_.lower() for x in prefix_nlp if x.pos_ in ["PROPN", "NUM", "NOUN"]])
                best_ents = " ".join([x.lemma_.lower() for x in best_nlp if x.pos_ in ["PROPN", "NUM", "NOUN"]])

                token_overlaps_ents["best"].append(
                    f1_score(best_ents, prefix_ents, stopwords=stopwords.words('english'), gram=args.gram)[0]
                )

        print(f"Results for {file}...")
        print(f"Best gen num tokens = {np.mean(num_tokens)}")
        print(f"Random gen num tokens = {np.mean(num_tokens_random)}")
        print(f"Human token overlap = {np.mean(token_overlaps['human']):.3f}")
        print(f"Random token overlap = {np.mean(token_overlaps['random']):.3f}")
        print(f"Best gen token overlap = {np.mean(token_overlaps['best']):.3f}")
        print(f"Best gen token overlap entities = {np.mean(token_overlaps_ents['best']):.3f}")

        print(f"Human rep = {np.mean(rep_scores['human']):.3f}")
        print(f"Random rep = {np.mean(rep_scores['random']):.3f}")
        print(f"Best gen rep = {np.mean(rep_scores['best']):.3f}")

        print(f"Gold beats generation = {np.mean(gold_beats_gen)}")

        latex_token_overlap.append(np.mean(token_overlaps['best']))
        latex_rep_score.append(np.mean(rep_scores['best']))

        random_latex_token_overlap.append(np.mean(token_overlaps['random']))
        random_latex_rep_score.append(np.mean(rep_scores['random']))

        latex_gold_beats_gen.append(np.mean(gold_beats_gen))

        if args.eval_pos_overlap:
            latex_token_overlap_ents.append(np.mean(token_overlaps_ents['best']))
            with open(file + ".ent_overlap.pkl", "wb") as f:
                pickle.dump(token_overlaps_ents, f)


        if i == 0 and args.eval_mauve:
            if os.path.exists(file + ".mauve.pkl"):
                with open(file + ".mauve.pkl", "rb") as f:
                    mauve_data = pickle.load(f)
                    mauve2 = mauve_data["max_gen_mauve"]
                    if "random_gen_mauve" in mauve_data:
                        mauve1 = mauve_data["random_gen_mauve"]
                    else:
                        mauve1 = None
            else:
                mauve1 = None
                mauve2 = mauve.compute_mauve(p_text=all_max_score, q_text=all_human, device_id=0, max_text_length=768, verbose=False)
            print(f"Max score mauve = {mauve2.mauve}")
            latex_mauve.append(mauve2.mauve)

        if args.eval_mauve and args.eval_type == "both" and mauve1 is None:
            mauve1 = mauve.compute_mauve(p_text=all_gen, q_text=all_human, device_id=0, max_text_length=768, verbose=False)

        if args.eval_mauve and mauve1 is not None:
            print(f"Random gen mauve = {mauve1.mauve}")

        if i == 0 and args.plot_divergence:
            plt.rcParams.update({'font.size': 16})
            plt.axis([0.0, 1.0, 0.0, 1.0])
            plt.plot(mauve1.divergence_curve[:, 0], mauve1.divergence_curve[:, 1])
            plt.plot(mauve2.divergence_curve[:, 0], mauve2.divergence_curve[:, 1])
            plt.fill_between(mauve1.divergence_curve[:, 0], mauve1.divergence_curve[:, 1], hatch='o', label="Nucleus", facecolor='white', edgecolor=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
            plt.fill(np.append(mauve1.divergence_curve[:, 0], mauve2.divergence_curve[:, 0][::-1]),
                     np.append(mauve1.divergence_curve[:, 1], mauve2.divergence_curve[:, 1][::-1]),
                     hatch='/', label="RankGen", facecolor='white', edgecolor=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
            #         fancybox=True, shadow=False, ncol=1)
            plt.legend(loc='upper right')
            plt.xlabel("similarity to Q")
            plt.ylabel("similarity to P")
            plt.savefig(f'{file}.plot.pdf', bbox_inches="tight")
            plt.clf()

        if args.eval_mauve:
            outputs = {
                "max_gen_mauve": mauve2,
                "random_gen_mauve": mauve1
            }
            with open(file + ".mauve.pkl", "wb") as f:
                pickle.dump(outputs, f)

print("Random gen latex --- ")
print(f"Latex token overlap = {' & '.join([f'{100 * x:.1f}' for x in random_latex_token_overlap])} & {100 * np.mean(random_latex_token_overlap):.1f}")
# print(f"Latex token overlap entities = {' & '.join([f'{100 * x:.1f}' for x in random_latex_token_overlap_ents])} & {100 * np.mean(random_latex_token_overlap_ents):.1f}")
print(f"Latex rep = {' & '.join([f'{100 * x:.1f}' for x in random_latex_rep_score])} & {100 * np.mean(random_latex_rep_score):.1f}")


print("Best gen latex --- ")
print(f"Latex token overlap = {' & '.join([f'{100 * x:.1f}' for x in latex_token_overlap])} & {100 * np.mean(latex_token_overlap):.1f}")
print(f"Latex token overlap entities = {' & '.join([f'{100 * x:.1f}' for x in latex_token_overlap_ents])} & {100 * np.mean(latex_token_overlap_ents):.1f}")
print(f"Latex rep = {' & '.join([f'{100 * x:.1f}' for x in latex_rep_score])} & {100 * np.mean(latex_rep_score):.1f}")

print(f"Gold beats gen latex = {' & '.join([f'{100 * x:.1f}' for x in latex_gold_beats_gen])} & {100 * np.mean(latex_gold_beats_gen):.1f}")

print(" ")
print(f"Mauve latex = {' & '.join([f'{x:.3f}' for x in latex_mauve])} & {np.mean(latex_mauve):.3f}")
