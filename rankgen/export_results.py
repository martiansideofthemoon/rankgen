import argparse
import json
import csv
import os


parser = argparse.ArgumentParser()
parser.add_argument('--best', default="data/wiki_beam_2_num_tokens_20_num_samples_10.jsonl")
parser.add_argument('--nucleus', default="data/wiki_beam_1_num_tokens_128_num_samples_1.jsonl")
parser.add_argument('--folder', default="analyze_results/t5_xxl_descartes")
parser.add_argument('--num_instances', default=50)
args = parser.parse_args()

os.makedirs(args.folder, exist_ok=True)

with open(args.best, "r") as f:
    best = [json.loads(x) for x in f.read().strip().split("\n")]

with open(args.nucleus, "r") as f:
    nuc = [json.loads(x) for x in f.read().strip().split("\n")]

output = [["Prefix", "Gold Suffix", "Best Suffix", "Nucleus Suffix", "Folder"]]

for b in best[:args.num_instances]:
    for n in nuc:
        if n['prefix'] == b['prefix']:
            output.append([
                b["prefix"], b["targets"][0], b['t5_xxl_descartes_outputs'][0], n['t5_xxl_descartes_outputs'][0], args.folder
            ])

with open(args.folder + "/outputs.csv", 'w', newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(output)
