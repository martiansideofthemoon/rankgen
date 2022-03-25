import argparse
import glob
from lib2to3.pgen2 import token
import numpy as np
import tqdm
import json
import torch
import os

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils import execute_gpt2, cudafy_tokens

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="data/t5_xl_all_domains_pg19_random.jsonl")
parser.add_argument('--model_size', default="medium")
args = parser.parse_args()

with open(args.dataset, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

tokenizer = GPT2Tokenizer.from_pretrained(f"gpt2-{args.model_size}")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(f"gpt2-{args.model_size}")
model.cuda()
model.eval()

avg_score = []
all_score = []

for idx, dd in tqdm.tqdm(enumerate(data), total=len(data)):
    prefix = dd['prefix']
    candidates = [dd['suffix']] + dd['negatives']
    assert len(candidates) == 11
    sequences = [prefix.strip() + " " + x.strip() for x in candidates]
    with torch.no_grad():
        inputs = cudafy_tokens(tokenizer(sequences, return_tensors="pt", padding=True, truncation=True))
        outputs = model(**inputs)
        out_log_probs = torch.nn.functional.log_softmax(outputs["logits"], dim=-1)
        gold_log_probs = torch.gather(out_log_probs[:, :-1, :], 2, inputs['input_ids'][:, 1:].unsqueeze(-1)).squeeze()
        token_mask = inputs['input_ids'][:, 1:] != tokenizer.pad_token_id
        gold_log_probs = gold_log_probs * token_mask
        perplexities = torch.exp(-1 * gold_log_probs.sum(dim=1) / token_mask.sum(dim=1))
        perplexities = perplexities.cpu().tolist()
        avg_score.append(np.mean([perplexities[0] < y for y in perplexities[1:]]))
        all_score.append(all([perplexities[0] < y for y in perplexities[1:]]))

    if (idx + 1) % 100 == 0:
        print(f"{np.mean(avg_score):.4f} average ({len(avg_score)} instances), {np.mean(all_score):.4f} all ({len(all_score)} instances)")

print(f"{np.mean(avg_score):.4f} average ({len(avg_score)} instances), {np.mean(all_score):.4f} all ({len(all_score)} instances)")
