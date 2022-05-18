import argparse
import json
import random
import random
import numpy as np
import mauve
import glob
import os
import tqdm
import pickle
from nltk import tokenize
import spacy
from nltk.corpus import stopwords
from utils import f1_score, rep_statistic

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="data/multi_outs/t5_xxl_descartes_wiki_greedy.tsv")
parser.add_argument('--num_samples', default=None, type=int)
parser.add_argument('--num_runs', default=1, type=int)
parser.add_argument('--rep_window', default=20, type=int)
parser.add_argument('--num_instances', default=7713, type=int)
parser.add_argument('--eval_mauve', action='store_true')
parser.add_argument('--eval_pos_overlap', action='store_true')
args = parser.parse_args()


files = glob.glob(args.dataset)

base_dir = os.path.dirname(files[0])
assert all([os.path.dirname(x) == base_dir for x in files])
files = ['pg19_gpt2_medium.tsv', 'wiki_gpt2_medium.tsv', 'pg19_gpt2_xl.tsv', 'wiki_gpt2_xl.tsv',
            'pg19_t5_xxl.tsv', 'wiki_t5_xxl.tsv', 'pg19_t5_xxl_descartes.tsv', 'wiki_t5_xxl_descartes.tsv']
files = [os.path.join(base_dir, f) for f in files]

latex_token_overlap = []
latex_rep_score = []
latex_token_overlap_ents = []
latex_mauve = []


if args.eval_pos_overlap:
    nlp = spacy.load("en_core_web_sm")

for file in files:
    if not os.path.exists(file):
        continue
    print(file)
    with open(file, 'r') as f:
        data = [x.split('\t') for x in f.read().strip().split("\n")]
        data_dict = {dd[0]: dd[1] for dd in data}

    if "wiki_" in file:
        with open("data_new/ppl/wiki_t5_xxl.jsonl", "r") as f:
            raw_inp_data = [json.loads(x) for x in f.read().strip().split("\n")]
        for rid in tqdm.tqdm(raw_inp_data):
            assert rid["prefix"] in data_dict
    elif "pg19_" in file:
        with open("data_new/ppl/pg19_t5_xxl.jsonl", "r") as f:
            raw_inp_data = [json.loads(x) for x in f.read().strip().split("\n")]
        for rid in tqdm.tqdm(raw_inp_data):
            assert rid["prefix"] in data_dict


    if args.num_samples is None:
        args.num_samples = (len(data) // args.num_instances) - 1

    assert len(data) % (args.num_samples + 1) == 0

    token_overlaps = {
        "human": [],
        "random": []
    }
    token_overlaps_ents = {
        "human": [],
        "random": []
    }
    rep_scores = {
        "human": [],
        "random": []
    }

    all_mauve = []
    for idx in range(args.num_runs):
        all_human = []
        all_gen = []
        for i in tqdm.tqdm(range(0, len(data), args.num_samples + 1)):
            gen_suffices = []
            for j in range(1, args.num_samples + 1):
                assert data[i][0] == data[i + j][0]
                gen_suffices.append(data[i + j][1])

            random_gen = random.choice(gen_suffices)
            all_human.append(data[i][0] + ' ' + data[i][1])
            all_gen.append(data[i][0] + ' ' + random_gen)

            token_overlaps["human"].append(
                f1_score(data[i][1], data[i][0], stopwords=stopwords.words('english'))[0]
            )
            token_overlaps["random"].append(
                f1_score(random_gen, data[i][0], stopwords=stopwords.words('english'))[0]
            )
            rep_scores["human"].append(rep_statistic(data[i][0], data[i][1], window=args.rep_window))
            rep_scores["random"].append(rep_statistic(data[i][0], random_gen, window=args.rep_window))

            if args.eval_pos_overlap and not os.path.exists(file + ".ent_overlap.pkl"):
                prefix_nlp = nlp(data[i][0])
                best_nlp = nlp(random_gen)
                prefix_ents = " ".join([x.lemma_.lower() for x in prefix_nlp if x.pos_ in ["PROPN", "NUM", "NOUN"]])
                best_ents = " ".join([x.lemma_.lower() for x in best_nlp if x.pos_ in ["PROPN", "NUM", "NOUN"]])

                token_overlaps_ents["random"].append(
                    f1_score(best_ents, prefix_ents, stopwords=stopwords.words('english'))[0]
                )

        print(f"Results for {file}...")
        print(f"token overlap = {np.mean(token_overlaps['human']):.3f} human, {np.mean(token_overlaps['random']):.3f} random")
        print(f"rep = {np.mean(rep_scores['human']):.3f} human, {np.mean(rep_scores['random']):.3f} random")

        latex_token_overlap.append(np.mean(token_overlaps['random']))
        latex_rep_score.append(np.mean(rep_scores['random']))

        if args.eval_pos_overlap and not os.path.exists(file + ".ent_overlap.pkl"):
            latex_token_overlap_ents.append(np.mean(token_overlaps_ents['random']))
            with open(file + ".ent_overlap.pkl", "wb") as f:
                pickle.dump(token_overlaps_ents, f)
        else:
            with open(file + ".ent_overlap.pkl", "rb") as f:
                token_overlaps_ents = pickle.load(f)
            latex_token_overlap_ents.append(np.mean(token_overlaps_ents['random']))

        if args.eval_mauve:
            mauve_file = file + ".mauve.pkl"
            if idx > 0:
                mauve_file += f".{idx}"
            if os.path.exists(mauve_file):
                with open(mauve_file, "rb") as f:
                    mauve_data = pickle.load(f)
                    mauve1 = mauve_data["max_gen_mauve"]
            else:
                mauve1 = mauve.compute_mauve(p_text=all_gen, q_text=all_human, device_id=0, max_text_length=768, verbose=False)
                outputs = {
                    "max_gen_mauve": mauve1
                }
                with open(mauve_file, "wb") as f:
                    pickle.dump(outputs, f)
            # print(mauve1.mauve)
            all_mauve.append(mauve1.mauve)

    if args.eval_mauve:
        print(np.mean(all_mauve))
        latex_mauve.append(np.mean(all_mauve))

print(f"Latex token overlap = {' & '.join([f'{100 * x:.1f}' for x in latex_token_overlap])} & {np.mean(latex_token_overlap):.3f}")
print(f"Latex rep = {' & '.join([f'{100 * x:.1f}' for x in latex_rep_score])} & {np.mean(latex_rep_score):.3f}")
print(f"Latex token overlap ents = {' & '.join([f'{100 * x:.1f}' for x in latex_token_overlap_ents])} & {np.mean(latex_token_overlap_ents):.3f}")
print(f"Mauve latex = {' & '.join([f'{x:.3f}' for x in latex_mauve])} & {np.mean(latex_mauve):.3f}")
