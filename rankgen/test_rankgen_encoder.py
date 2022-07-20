import argparse
import json
import torch
import tqdm
import os
import numpy as np
import time
from rankgen import RankGenEncoder, RankGenGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="kalpeshk2011/rankgen-t5-base-all", type=str)

parser.add_argument('--cache_dir', default=None, type=str)
args = parser.parse_args()

test_example_file_map = {
    "kalpeshk2011/rankgen-t5-base-all": "rankgen_data/test_examples/t5_base_all.jsonl",
    "kalpeshk2011/rankgen-t5-large-all": "rankgen_data/test_examples/t5_large_all.jsonl",
    "kalpeshk2011/rankgen-t5-xl-all": "rankgen_data/test_examples/t5_xl_all.jsonl",
    "kalpeshk2011/rankgen-t5-xl-pg19": "rankgen_data/test_examples/t5_xl_pg19.jsonl"
}

rankgen_encoder = RankGenEncoder(args.model_path)

parameters = sum(p.numel() for p in rankgen_encoder.model.parameters())

f = open(test_example_file_map[args.model_path], "r")
examples = [json.loads(x) for x in f.read().strip().split("\n")]

mean_prefix_diff = []
mean_suffix_diff = []

start = time.time()
all_prefix_outs = rankgen_encoder.encode([x["inputs"]["inputs_pretokenized"] for x in examples], vectors_type="prefix", verbose=True, return_input_ids=True)
all_suffix_outs = rankgen_encoder.encode([x["inputs"]["targets_pretokenized"] for x in examples], vectors_type="suffix", verbose=True, return_input_ids=True)
time_taken = time.time() - start

print(f"Time taken = {time_taken / len(examples)}")

for eg_num, eg in tqdm.tqdm(enumerate(examples)):
    ref_prefix_vec = torch.Tensor(eg['score']['input_embedding']).cuda()
    ref_suffix_vec = torch.Tensor(eg['score']['target_embedding']).cuda()
    ref_prefix_ids = eg['inputs_processed']['prefix_ids']
    ref_suffix_ids = eg['inputs_processed']['suffix_ids']

    mean_prefix_diff.append(torch.mean(torch.abs(all_prefix_outs['embeddings'][eg_num] - ref_prefix_vec)).item())
    mean_suffix_diff.append(torch.mean(torch.abs(all_suffix_outs['embeddings'][eg_num] - ref_suffix_vec)).item())

    for x, y in zip(ref_prefix_ids, all_prefix_outs['input_ids'][eg_num]):
        assert x == y

    for x, y in zip(ref_suffix_ids, all_suffix_outs['input_ids'][eg_num]):
        assert x == y

# Expected to be close to 10e-3
print(np.mean(mean_prefix_diff))
print(np.mean(mean_suffix_diff))
