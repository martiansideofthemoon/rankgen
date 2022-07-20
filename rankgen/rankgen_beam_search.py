from sys import prefix
from transformers import T5Tokenizer, T5EncoderModel
import pickle
import argparse
import numpy as np
import tqdm
import os
import torch
import random
import json
from functools import partial
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils import form_partitions
from rankgen import RankGenEncoder, RankGenGenerator
from utils import truncate
from transformers.utils import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="rankgen_data/wiki.jsonl", type=str)
parser.add_argument('--num_samples', default=10, type=int)
parser.add_argument('--beam_size', default=2, type=int)
parser.add_argument('--num_tokens', default=20, type=int)
parser.add_argument('--max_length', default=115, type=int)
parser.add_argument('--top_p', default=0.9, type=float)
parser.add_argument('--model_size', default='medium', type=str)
parser.add_argument('--cache_dir', default=None, type=str)
parser.add_argument('--rankgen_encoder', default='kalpeshk2011/rankgen-t5-xl-all', type=str)
parser.add_argument('--num_shards', default=1, type=int)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--output_file', default=None, type=str)
args = parser.parse_args()

with open(args.dataset, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

if args.num_shards > 1:
    partitions = form_partitions(data, args.num_shards)
    data = partitions[args.local_rank]
    args.output_file = f'{args.output_file}.shard_{args.local_rank}'

rankgen_encoder = RankGenEncoder(model_path=args.retriever_model_path, cache_dir=args.cache_dir)

random.seed(49)
random.shuffle(data)

random.seed(442)
random.shuffle(data)

folder_name = f"token_bs_t5x"

rankgen_generator = RankGenGenerator(rankgen_encoder=rankgen_encoder, language_model="gpt2-{args.model_size}", cache_dir=args.cache_dir)

outputs = []

target_seq_len = []
gen_seq_len = []

logging.set_verbosity_error()

if os.path.exists(args.output_file):
    with open(args.output_file, "r") as f:
        outputs = f.read().strip().split("\n")

for kk, instance in tqdm.tqdm(enumerate(data), total=len(data)):
    if kk < len(outputs):
        continue
    token_beam_text, token_beam_scores = rankgen_generator.beam_search(contexts=[instance["prefix"]],
                                                                       beam_size=args.beam_size,
                                                                       top_p=args.top_p,
                                                                       num_tokens=args.num_tokens,
                                                                       num_samples=args.num_samples,
                                                                       max_length=args.max_length)

    token_beam_text = token_beam_text[0]
    token_beam_text = [truncate(" ".join(x.split())) for x in token_beam_text]
    if "scores" not in instance:
        instance["scores"] = [1.0]
    outputs.append(json.dumps({
        "prefix": instance["prefix"],
        "targets": instance["targets"][0:1] + token_beam_text,
        "scores": instance["scores"][0:1] + token_beam_scores[0].cpu().tolist()
    }))
    target_seq_len.append(len(instance["targets"][0].split()))
    gen_seq_len.append(len(token_beam_text[0].split()))

    if (kk + 1) % 100 == 0:
        print(f"Avg lens ({kk + 1} instances) = {np.mean(gen_seq_len)} generation, {np.mean(target_seq_len)} target")
        print("Saving file...")
        with open(args.output_file, "w") as f:
            f.write("\n".join(outputs) + "\n")

with open(args.output_file, "w") as f:
    f.write("\n".join(outputs) + "\n")
