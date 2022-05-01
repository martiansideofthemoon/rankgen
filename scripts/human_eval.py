import argparse
import json
import csv
import os
import random


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="data/t5_xl_all_domains_wiki_hard.jsonl")
parser.add_argument('--folder', default="ab_tests/gold_neg_wiki_hard")
parser.add_argument('--method', default="nucleus")
parser.add_argument('--num_instances', default=200)
args = parser.parse_args()

os.makedirs(args.folder, exist_ok=True)

with open(args.dataset, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

random.seed(43)

output = [["Prefix", "First", "Second", "Order", "InstanceNum", "Folder"]]

for i, dd in enumerate(data[:args.num_instances]):
    negative = random.choice(dd["negatives"])
    order = random.random()
    print(dd.keys())
    if args.method == "nucleus":
        if order < 0.5:
            output.append([
                dd["prefix"], dd["suffix"], negative, "beam,nucleus", i, args.folder
            ])
        else:
            output.append([
                dd["prefix"], negative, dd["suffix"], "nucleus,beam", i, args.folder
            ])
    elif args.method == "beam":
        if order < 0.5:
            output.append([
                dd["prefix"], dd["suffix"], negative, "beam,nucleus", i, args.folder
            ])
        else:
            output.append([
                dd["prefix"], negative, dd["suffix"], "nucleus,beam", i, args.folder
            ])

with open(args.folder + "/input.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(output)