import argparse
import json
from tabnanny import verbose
import torch
import tqdm
import os
import numpy as np
import time
from t5x_embeddings import T5XEmbeddingGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default="t5x_conversion/t5_large_all", type=str)
parser.add_argument('--cache_dir', default=None, type=str)
args = parser.parse_args()


t5x_embedder = T5XEmbeddingGenerator(model_path=args.model_path, cache_dir=None)

f = open(os.path.join(args.model_path, "examples.jsonl"), "r")
examples = [json.loads(x) for x in f.read().strip().split("\n")]

mean_prefix_diff = []
mean_suffix_diff = []

start = time.time()
all_prefix_outs = t5x_embedder.encode([x["inputs"]["inputs_pretokenized"] for x in examples],  vectors_type="prefix", verbose=True, return_input_ids=True)
all_suffix_outs = t5x_embedder.encode([x["inputs"]["targets_pretokenized"] for x in examples],  vectors_type="suffix", verbose=True, return_input_ids=True)
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
