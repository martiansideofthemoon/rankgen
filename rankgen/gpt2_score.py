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
parser.add_argument('--output_path', default=None)
parser.add_argument('--model_size', default="medium")
parser.add_argument('--num_negatives', default=20, type=int)
args = parser.parse_args()

tokenizer = GPT2Tokenizer.from_pretrained(f"gpt2-{args.model_size}")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(f"gpt2-{args.model_size}")
model.cuda()
model.eval()

avg_score = []
all_score = []

def compute_gpt2(sequences):
    with torch.no_grad():
        inputs = cudafy_tokens(tokenizer(sequences, return_tensors="pt", padding=True, truncation=True))
        outputs = model(**inputs)
        out_log_probs = torch.nn.functional.log_softmax(outputs["logits"], dim=-1)
        gold_log_probs = torch.gather(out_log_probs[:, :-1, :], 2, inputs['input_ids'][:, 1:].unsqueeze(-1)).squeeze()
        token_mask = inputs['input_ids'][:, 1:] != tokenizer.pad_token_id
        gold_log_probs = gold_log_probs * token_mask
        perplexities = torch.exp(-1 * gold_log_probs.sum(dim=1) / token_mask.sum(dim=1))
        perplexities = perplexities.cpu().tolist()
    return perplexities


if args.dataset.endswith(".jsonl"):
    with open(args.dataset, "r") as f:
        data = [json.loads(x) for x in f.read().strip().split("\n")]

    if os.path.exists(args.output_path):
        with open(args.output_path, "r") as f:
            outputs = [x for x in f.read().strip().split("\n")]
    else:
        outputs = []

    for idx, dd in tqdm.tqdm(enumerate(data), total=len(data)):
        if idx < len(outputs):
            continue
        prefix = dd['prefix']
        if 'targets' in dd:
            candidates = dd['targets']
        else:
            candidates = [dd['suffix']] + dd['negatives']
        assert len(candidates) == args.num_negatives + 1
        sequences = [prefix.strip() + " " + x.strip() for x in candidates]
        perplexities = compute_gpt2(sequences)
        avg_score.append(np.mean([perplexities[0] < y for y in perplexities[1:]]))
        all_score.append(all([perplexities[0] < y for y in perplexities[1:]]))

        if (idx + 1) % 100 == 0:
            print(f"{np.mean(avg_score):.4f} average ({len(avg_score)} instances), {np.mean(all_score):.4f} all ({len(all_score)} instances)")

        outputs.append(json.dumps({
            "prefix": prefix,
            "targets": candidates,
            "scores": [-1 * x for x in perplexities]
        }))

        if idx % 100 == 0:
            with open(args.output_path, "w") as f:
                f.write("\n".join(outputs) + "\n")

    with open(args.output_path, "w") as f:
        f.write("\n".join(outputs) + "\n")

elif args.dataset.endswith(".tsv"):
    with open(args.dataset, "r") as f:
        data = [x.split("\t") for x in f.read().strip().split("\n")]

    outputs = []
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = args.dataset + ".ppl_scores"

    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            outputs = [x for x in f.read().strip().split("\n")]
    for idx in tqdm.tqdm(range(len(outputs) * (args.num_negatives + 1), len(data), args.num_negatives + 1)):
        prefix = data[idx][0]
        candidates = []
        for jdx in range(args.num_negatives + 1):
            assert data[idx + jdx][0] == prefix
            candidates.append(data[idx + jdx][1])
        assert len(candidates) == args.num_negatives + 1
        sequences = [prefix.strip() + " " + x.strip() for x in candidates]
        perplexities = [compute_gpt2(x)[0] for x in sequences]
        avg_score.append(np.mean([perplexities[0] < y for y in perplexities[1:]]))
        all_score.append(all([perplexities[0] < y for y in perplexities[1:]]))

        assert len(candidates) == len(perplexities)
        outputs.append(json.dumps({
            "prefix": prefix,
            "targets": candidates,
            "scores": [-1 * x for x in perplexities]
        }))

        if len(avg_score) % 100 == 0:
            print(f"{np.mean(avg_score):.4f} average ({len(avg_score)} instances), {np.mean(all_score):.4f} all ({len(all_score)} instances)")
            with open(output_path, "w") as f:
                f.write("\n".join(outputs) + "\n")

    with open(output_path, "w") as f:
        f.write("\n".join(outputs) + "\n")

print(f"{np.mean(avg_score):.4f} average ({len(avg_score)} instances), {np.mean(all_score):.4f} all ({len(all_score)} instances)")
