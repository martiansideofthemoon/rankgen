import argparse
import collections
import json
import csv
import os
import glob
import numpy as np
import pickle
import random
from regex import E
import tqdm
from datetime import datetime
from collections import Counter, defaultdict
from scipy.stats import kendalltau
from statsmodels.stats.inter_rater import fleiss_kappa


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="human-eval-data/*")
parser.add_argument('--split', default=None)
parser.add_argument('--model', default=None)
args = parser.parse_args()

def get_annotation(hit):
    text = hit[33]
    if text.strip().lower().startswith("text 1"):
        return "text 1"
    if text.strip().lower().startswith("test 1"):
        return "text 1"
    elif text.strip().lower().startswith("text 2"):
        return "text 2"
    elif text.startswith("Neither of them sound like a good continuation to me, but I choose Text 1"):
        return "text 1"
    elif text == "Xcdsds" or text == "adfwegsdgsdgsdg":
        return None
    else:
        import pdb; pdb.set_trace()
        pass

def get_multi_annotations(hits):
    anns = [get_annotation(x) for x in hits]
    return [x for x in anns if x]

def most_common(lst):
    return max(set(lst), key=lst.count)

def print_counter(x):
    x = Counter(x)
    total = sum([v for v in x.values()])
    for k, v in x.items():
        print(f"{k} = {v / total:.4f} ({v} / {total})")

data = []
header = None
files = glob.glob(args.dataset)
for fl in files:
    curr_data = []
    with open(fl, "r") as f:
        spamreader = csv.reader(f)
        for row in spamreader:
            curr_data.append(row)
    header = curr_data[0]
    data.extend(curr_data[1:])

if args.split is not None:
    data = [x for x in data if x[31] == args.split]

if args.model is not None:
    data = [x for x in data if x[32] == args.model]

worker_ids = list(set([x[15] for x in data]))
worker_ids.sort()
hit_ids = list(set([x[0] for x in data]))
scarecrow_beam = defaultdict(list)

all_workers = []
for worker in worker_ids:
    verdicts = []
    data_small = [x for x in data if x[15] == worker]
    for dd in data_small:
        annotation = get_annotation(dd)
        text1, text2 = dd[30].split(",")
        if annotation == "text 1":
            verdicts.append(text1)
        elif annotation == "text 2":
            verdicts.append(text2)

        if annotation is not None and verdicts[-1] == "beam":
            # compute scarecrow stats for cases beam search triumphs
            scarecrow_beam[worker].append(dd[34])

    all_workers.extend(verdicts)
    if verdicts:
        print(f"{worker} results:")
        print(Counter(verdicts))


# Agreement between annotators
annotations = []
unique = []
table = []
majority = []
for hit_id in hit_ids:
    curr_entry = [0, 0]
    data_small = [x for x in data if x[0] == hit_id]
    workers = [x[15] for x in data_small]
    text1, text2 = data_small[0][30].split(",")

    anns = get_multi_annotations(data_small)

    unique.append(len(set(anns)))

    for ann in anns[:3]:
        if ann == "text 1":
            curr_entry[0] += 1
        elif ann == "text 2":
            curr_entry[1] += 1

    vote = most_common(anns)
    if vote == "text 1":
        majority.append(text1)
    elif vote == "text 2":
        majority.append(text2)
    table.append(curr_entry)


table = np.array(table)

print("")

print(f"Fleiss ({len(table)} pairs) = {fleiss_kappa(table)}")
print_counter(unique)
print("")

print("Majority vote accuracy ---")
print_counter(majority)
print("")

print("Absolute accuracy ---")
print_counter(all_workers)
print("")

# Scarecrow statistics

def process_scarecrow(sc_anns):
    totals = defaultdict(int)
    for sca in sc_anns:
        types = [x.strip() for x in sca.split(", ")]
        types = [x for x in types if x != "equal" and x.strip()]
        for tp in types:
            totals[tp] += 1 / len(types)
    return totals

scarecrow_list = []
for k, v in scarecrow_beam.items():
    scarecrow_list.extend(v)

print("All annotators --- ")
scarecrow_list = [x for x in scarecrow_list if x]
scarecrow_all = process_scarecrow(scarecrow_list)
for k, v in scarecrow_all.items():
    print(f"{k} = {v * 100 / len(scarecrow_list):.1f}")
