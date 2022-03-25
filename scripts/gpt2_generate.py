import argparse
import glob
from lib2to3.pgen2 import token
import numpy as np
import tqdm
import json
import torch
import os
import random
import nltk

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils import execute_gpt2, cudafy_tokens

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="data/t5_xl_all_domains_wiki_random.jsonl")
parser.add_argument('--output_file', default="data/wiki_gpt2_medium_p90.tsv")
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
random.seed(43)
device = "cuda" if torch.cuda.is_available() else "cpu"

def postprocess(outputs):
    return "".join(tokenizer.batch_decode(outputs, skip_special_tokens=True))

output = ""
suffix_lens = []
gen_lens = []

def truncate(text):
    last_punc = 0
    if "." in text:
        last_punc = max(last_punc, text.rindex("."))
    if "?" in text:
        last_punc = max(last_punc, text.rindex("?"))
    if "!" in text:
        last_punc = max(last_punc, text.rindex("!"))
    if last_punc != 0:
        text = text[:last_punc + 1]
    return text


for idx, dd in tqdm.tqdm(enumerate(data), total=len(data)):
    prefix = dd['prefix']
    batch = tokenizer(prefix, truncation=True, padding="longest", return_tensors="pt").to(device)
    num_tokens = len(batch['input_ids'][0])
    with torch.no_grad():
        generation = model.generate(**batch, do_sample=True, output_scores=True, return_dict_in_generate=True, max_new_tokens=128, top_p=0.9)
        gen_text = postprocess(generation['sequences'][0, num_tokens:])
        gen_text = " ".join(gen_text.split())
        gen_text = truncate(gen_text)

    if random.random() < 0.5:
        gen_text = truncate(gen_text[:-1])

    suffix_lens.append(len(dd['suffix'].split()))
    gen_lens.append(len(gen_text.split()))
    output += f"{prefix}\t{dd['suffix']}\tplaceholder\tplaceholder\n"
    output += f"{prefix}\t{gen_text}\tplaceholder\tplaceholder\n"

    if (idx + 1) % 100 == 0:
        print(f"Avg suffix length = {np.mean(suffix_lens):.4f}, avg gen length = {np.mean(gen_lens):.4f},")
        with open(args.output_file, "w") as f:
            f.write(output)
