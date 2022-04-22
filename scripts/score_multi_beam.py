import argparse
import json
import random
import numpy as np
import mauve
import pickle
import os
from utils import truncate

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="outputs_beam/wiki_t5_large_beam_1_tokens_115_samples_1.jsonl")
parser.add_argument('--domain', default="wiki")
parser.add_argument('--gen_key_type', default=None)
parser.add_argument('--data_length', default=7713, type=int)
args = parser.parse_args()

with open(args.dataset, 'r') as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

data_dict = {x["prefix"]: x for x in data}

assert len(data) == args.data_length
assert len(data_dict) == args.data_length

if args.domain == "wiki":
    with open("data/multi_outs/t5_xxl_descartes_wiki_ppl.jsonl", "r") as f:
        raw_inp_data = [json.loads(x) for x in f.read().strip().split("\n")]
    for rid in raw_inp_data:
        assert rid["prefix"] in data_dict
        assert rid["targets"][0] == data_dict[rid["prefix"]]["targets"][0]
else:
    with open("data_new/ppl/pg19_t5_xxl.jsonl", "r") as f:
        raw_inp_data = [json.loads(x) for x in f.read().strip().split("\n")]
    for rid in raw_inp_data:
        assert rid["prefix"] in data_dict
        assert rid["targets"][0] == data_dict[rid["prefix"]]["targets"][0]

all_human = []
all_gen = []
num_tokens = []
for dd in data:
    all_human.append(dd['prefix'] + ' ' + dd['targets'][0])
    if args.gen_key_type is None:
        # assert len(dd['targets']) != 21
        all_gen.append(dd['prefix'] + ' ' + dd['targets'][1])
        num_tokens.append(len(dd['targets'][1].split()))
    else:
        generation = dd[args.gen_key_type][0]
        generation = truncate(" ".join(generation.split()))
        num_tokens.append(len(generation.split()))
        all_gen.append(dd['prefix'] + ' ' + generation)

print(np.mean(num_tokens))
output_file = args.dataset + ".mauve.pkl"

if os.path.exists(output_file):
    with open(output_file, "rb") as f:
        mauve1 = pickle.load(f)["max_gen_mauve"]
    print(f"Generation score mauve = {mauve1.mauve}")
else:
    mauve1 = mauve.compute_mauve(p_text=all_gen, q_text=all_human, device_id=0, max_text_length=768, verbose=False)
    print(f"Generation score mauve = {mauve1.mauve}")
    outputs = {"max_gen_mauve": mauve1}

    with open(args.dataset + ".mauve.pkl", "wb") as f:
        pickle.dump(outputs, f)
