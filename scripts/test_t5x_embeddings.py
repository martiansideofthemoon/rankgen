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
parser.add_argument('--model_path', default="/data/kalpesh/t5x_conversion", type=str)
args = parser.parse_args()


t5x_embedder = T5XEmbeddingGenerator(model_path=args.model_path)

f = open(os.path.join(args.model_path, "pg19_debug_inference-score.jsonl-00000-of-00001.processed"), "r")
examples = [json.loads(x) for x in f.read().strip().split("\n")]

mean_prefix_diff = []
mean_suffix_diff = []

all_prefix_outs = t5x_embedder.encode([x["inputs"]["inputs_pretokenized"] for x in examples],  vectors_type="prefix", verbose=True)
all_suffix_outs = t5x_embedder.encode([x["inputs"]["targets_pretokenized"] for x in examples],  vectors_type="suffix", verbose=True)

for eg_num, eg in tqdm.tqdm(enumerate(examples)):
    ref_prefix_vec = torch.Tensor(eg['score']['input_embedding']).cuda()
    ref_suffix_vec = torch.Tensor(eg['score']['target_embedding']).cuda()
    ref_prefix_ids = eg['inputs_processed']['prefix_ids']
    ref_suffix_ids = eg['inputs_processed']['suffix_ids']

    mean_prefix_diff.append(torch.mean(torch.abs(all_prefix_outs['embeddings'][eg_num] - ref_prefix_vec)).item())
    mean_suffix_diff.append(torch.mean(torch.abs(all_suffix_outs['embeddings'][eg_num] - ref_suffix_vec)).item())

print(np.mean(mean_prefix_diff))
print(np.mean(mean_suffix_diff))
