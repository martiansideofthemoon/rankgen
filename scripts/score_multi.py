import argparse
import json
import random
import numpy as np
import mauve

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="data/multi_outs/t5_xxl_wiki_t5_xl_gen_inbook_all.jsonl")
args = parser.parse_args()

with open(args.dataset, 'r') as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

for i in range(5):
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

    if i != 0:
        mauve2 = mauve.compute_mauve(p_text=all_max_score, q_text=all_human, device_id=0, max_text_length=768, verbose=False)
        print(f"Max score mauve = {mauve2.mauve}")

    mauve1 = mauve.compute_mauve(p_text=all_gen, q_text=all_human, device_id=0, max_text_length=768, verbose=False)
    print(f"Random gen mauve = {mauve1.mauve}")
