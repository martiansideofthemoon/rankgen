import argparse
import json
import csv
import os
import numpy as np
import random
from utils import truncate


parser = argparse.ArgumentParser()
parser.add_argument('--folder', default="ab_tests/t5_xxl_descartes_nucleus_vs_beam")
parser.add_argument('--num_instances', default=50)
args = parser.parse_args()

BASE_DIR = "files_human_eval"

files = [
    # GPT2-medium
    # {"nucleus": ["pg19", "gpt2_medium", "pg19_gpt2_medium.jsonl"],
    #  "beam": ["pg19", "gpt2_medium", "pg19_t5_xl_beam_2_tokens_20_samples_10.jsonl"]},
    # {"nucleus": ["wiki", "gpt2_medium", "wiki_gpt2_medium.jsonl"],
    #  "beam": ["wiki", "gpt2_medium", "wiki_t5_xl_beam_2_tokens_20_samples_10.jsonl"]},
    # T5-XXL descartes
    {"nucleus": ["pg19", "t5_xxl_descartes", "pg19_t5_xxl_descartes.jsonl"],
     "beam": ["pg19", "t5_xxl_descartes", "pg19_beam_2_num_tokens_20_num_samples_10.jsonl"]},
    {"nucleus": ["wiki", "t5_xxl_descartes", "wiki_t5_xxl_descartes.jsonl"],
     "beam": ["wiki", "t5_xxl_descartes", "wiki_beam_2_num_tokens_20_num_samples_10.jsonl"]},
]

random.seed(43)

os.makedirs(args.folder, exist_ok=True)

output = [["Prefix", "First", "Second", "Order", "Dataset", "Model"]]
nucleus_tokens = []
beam_tokens = []

for file_pair in files:
    model = file_pair["nucleus"][1]
    dataset = file_pair["nucleus"][0]

    nucleus_file = file_pair["nucleus"][2]
    with open(f"{BASE_DIR}/{model}/{nucleus_file}", "r") as f:
        nucleus_data = [json.loads(x) for x in f.read().strip().split("\n")]

    beam_file = file_pair["beam"][2]
    with open(f"{BASE_DIR}/{model}/{beam_file}", "r") as f:
        beam_data = [json.loads(x) for x in f.read().strip().split("\n")]

    random.shuffle(nucleus_data)

    for i, nucleus_instance in enumerate(nucleus_data[:args.num_instances]):
        beam_instance = None
        for j, dd2 in enumerate(beam_data):
            if dd2['prefix'] == nucleus_instance['prefix']:
                beam_instance = dd2
                break

        assert beam_instance["prefix"] == nucleus_instance["prefix"]

        nucleus_gen = random.choice(nucleus_instance['targets'][1:])
        order = random.random()

        if model == "gpt2_medium":
            beam_gen = beam_instance["targets"][1]
        elif model == "t5_xxl_descartes":
            beam_gen = beam_instance["t5_xxl_descartes_outputs"][0]
            beam_gen = truncate(beam_gen)

        nucleus_tokens.append(len(nucleus_gen.split()))
        beam_tokens.append(len(beam_gen.split()))

        if order < 0.5:
            output.append([
                nucleus_instance["prefix"], beam_gen, nucleus_gen, "beam,nucleus", dataset, model
            ])
        else:
            output.append([
                nucleus_instance["prefix"], nucleus_gen, beam_gen, "nucleus,beam", dataset, model
            ])

with open(args.folder + "/input.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(output)

print(np.mean(beam_tokens))
print(np.mean(nucleus_tokens))
