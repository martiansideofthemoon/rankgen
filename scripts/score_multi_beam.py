import argparse
import json
import random
import numpy as np
import mauve

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="outputs_beam/wiki_t5_large_beam_1_tokens_115_samples_1.jsonl")
args = parser.parse_args()

with open(args.dataset, 'r') as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

all_human = []
all_gen = []
num_tokens = []
for dd in data:
    all_human.append(dd['prefix'] + ' ' + dd['targets'][0])
    all_gen.append(dd['prefix'] + ' ' + dd['targets'][1])

mauve1 = mauve.compute_mauve(p_text=all_gen, q_text=all_human, device_id=0, max_text_length=768, verbose=False)
print(f"Generation score mauve = {mauve1.mauve}")
