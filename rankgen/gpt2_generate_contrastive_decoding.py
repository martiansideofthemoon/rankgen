import argparse
import numpy as np
import tqdm
import json
import torch
import os
import random

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils import form_partitions, truncate

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="data/t5_xl_all_domains_wiki_random.jsonl")
parser.add_argument('--output_file', default="data_new/contrastive_decoding/wiki_gpt2_medium_ignore_prefix.tsv")
parser.add_argument('--model_size', default="medium")
parser.add_argument('--num_instances', default=7713, type=int)
parser.add_argument('--num_samples', default=1, type=int)
parser.add_argument('--max_new_tokens', default=115, type=int)
parser.add_argument('--top_k', default=5, type=int)
parser.add_argument('--penalty_alpha', default=0.6, type=float)
parser.add_argument('--truncate_fraction', default=0.0, type=float)
parser.add_argument('--num_shards', default=1, type=int)
parser.add_argument('--local_rank', default=0, type=int)
args = parser.parse_args()

with open(args.dataset, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

def ignore_prefix_prepare_inputs_for_generation(input_ids, past=None, **kwargs):
            
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    input_ids = input_ids[:, -1].unsqueeze(-1)
    if token_type_ids is not None:
        token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids[:, -1].unsqueeze(-1)
    else:
        position_ids = None

    return {
        "input_ids": input_ids,
        "past_key_values": past,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


student_lm = GPT2LMHeadModel.from_pretrained(f"gpt2")
tokenizer = GPT2Tokenizer.from_pretrained(f"gpt2-{args.model_size}")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(f"gpt2-{args.model_size}")
model.cuda()
model.eval()
student_lm.cuda()

student_lm.prepare_inputs_for_generation = ignore_prefix_prepare_inputs_for_generation

avg_score = []
all_score = []
random.seed(43)
device = "cuda" if torch.cuda.is_available() else "cpu"

output = ""
suffix_lens = []
gen_lens = []

def postprocess(outputs):
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def truncate(text):
    """Truncate text to the last full sentence."""
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

if args.num_shards > 1:
    partitions = form_partitions(data, args.num_shards)
    data = partitions[args.local_rank]
    args.output_file = f'{args.output_file}.shard_{args.local_rank}'

for idx, dd in tqdm.tqdm(enumerate(data), total=min(len(data), args.num_instances)):
    if len(suffix_lens) >= args.num_instances:
        break
    prefix = dd['prefix']
    batch = tokenizer(prefix, truncation=True, padding="longest", return_tensors="pt", max_length=1024 - args.max_new_tokens).to(device)
    num_tokens = len(batch['input_ids'][0])
    if num_tokens >= 1024 - args.max_new_tokens - 3:
        print("long sequence detected")

    with torch.inference_mode():
        generation = model.generate(
            **batch,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            min_prob=0.0,
            do_sample=False,
            num_beams=5,
            max_length=num_tokens + args.max_new_tokens,
            num_return_sequences=1,
            student_lm=student_lm,
            teacher_student=True,
            model_kwargs_student={}, 
            st_coef=1.0,
            tokenizer=tokenizer, # analysis
            student_min_prob=0.0,
            student_temperature=0.5,
            use_cap_student=False, #cap student debug
            use_switch=False
        )
        gen_text = postprocess(generation[:, num_tokens:])
        gen_text = [" ".join(x.split()) for x in gen_text]
        gen_text = [truncate(x) for x in gen_text]

    for i in range(len(gen_text)):
        if random.random() < args.truncate_fraction:
            gen_text[i] = truncate(gen_text[i][:-1])

    if "suffix" in dd:
        suffix_str = dd['suffix']
    else:
        suffix_str = dd['targets'][0]

    suffix_lens.append(len(suffix_str.split()))
    for x in gen_text:
        gen_lens.append(len(x.split()))
    output += f"{prefix}\t{suffix_str}\tplaceholder\tplaceholder\n"
    for x in gen_text:
        output += f"{prefix}\t{x}\tplaceholder\tplaceholder\n"

    if (idx + 1) % 100 == 0:
        print(f"Avg suffix length = {np.mean(suffix_lens):.4f} ({len(suffix_lens)} samples), avg gen length = {np.mean(gen_lens):.4f} ({len(gen_lens)} samples)")
        with open(args.output_file, "w") as f:
            f.write(output)

with open(args.output_file, "w") as f:
    f.write(output)
