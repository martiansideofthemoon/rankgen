import argparse
import json
import csv
import os
import random


parser = argparse.ArgumentParser()
parser.add_argument('--nucleus_dataset', default="/mnt/nfs/work1/miyyer/kalpesh/projects/presuf-retrieval/files_human_eval/pg19_gpt2_medium.jsonl")
parser.add_argument('--beam_dataset', default="/mnt/nfs/work1/miyyer/kalpesh/projects/presuf-retrieval/files_human_eval/pg19_t5_xl_beam_2_tokens_20_samples_10.jsonl")
parser.add_argument('--folder', default="ab_tests/pg19_nucleus_vs_beam")
parser.add_argument('--dataset', default="pg19")
parser.add_argument('--num_instances', default=200)
args = parser.parse_args()

os.makedirs(args.folder, exist_ok=True)

with open(args.nucleus_dataset, "r") as f:
    nucleus_data = [json.loads(x) for x in f.read().strip().split("\n")]
with open(args.beam_dataset, "r") as f:
    beam_data = [json.loads(x) for x in f.read().strip().split("\n")]

random.seed(43)

output = [["Prefix", "First", "Second", "Order", "NucleusInstanceNum", "BeamInstanceNum", "Dataset", "Folder", "NucleusPath", "BeamPath"]]

for i, n in enumerate(nucleus_data[:args.num_instances]):
    nucleus_gen = random.choice(n['targets'][1:])
    order = random.random()
    beam_instance_num = i
    for j, b in enumerate(beam_data):
        if b['prefix'] == n['prefix']:
            beam_gen = b['targets'][1]
        beam_instance_num = j
    if order < 0.5:
        output.append([
            n["prefix"], beam_gen, nucleus_gen, "beam,nucleus", i, beam_instance_num, args.dataset, args.folder, args.nucleus_dataset, args.beam_dataset
        ])
    else:
        output.append([
            n["prefix"], nucleus_gen, beam_gen, "nucleus,beam", i, beam_instance_num, args.dataset, args.folder, args.nucleus_dataset, args.beam_dataset
        ])

with open(args.folder + "/input.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(output)