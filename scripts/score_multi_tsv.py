import argparse
import json
import random
import random
import numpy as np
import mauve
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="data/multi_outs/t5_xxl_descartes_wiki_greedy.tsv")
parser.add_argument('--num_samples', default=1, type=int)
args = parser.parse_args()

with open(args.dataset, 'r') as f:
    data = [x.split('\t') for x in f.read().strip().split("\n")]

assert len(data) % (args.num_samples + 1) == 0

all_mauve = []
for _ in range(5):
    all_human = []
    all_gen = []
    for i in tqdm.tqdm(range(0, len(data), args.num_samples + 1)):
        gen_suffices = []
        for j in range(1, args.num_samples + 1):
            assert data[i][0] == data[i + j][0]
            gen_suffices.append(data[i + j][1])

        all_human.append(data[i][0] + ' ' + data[i][1])
        all_gen.append(data[i][0] + ' ' + random.choice(gen_suffices))

    mauve1 = mauve.compute_mauve(p_text=all_gen, q_text=all_human, device_id=0, max_text_length=768, verbose=False)
    print(mauve1.mauve)
    all_mauve.append(mauve1.mauve)

print(all_mauve)
print(np.mean(all_mauve))
