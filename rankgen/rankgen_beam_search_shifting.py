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
import nltk
from nltk.tokenize import sent_tokenize
from functools import partial
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils import form_partitions
from rankgen_encoder import RankGenEncoder
from utils import truncate
from transformers.utils import logging

nltk.download('punkt')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="data/multi_outs/t5_xxl_descartes_wiki_ppl.jsonl", type=str)
parser.add_argument('--num_samples', default=10, type=int)
parser.add_argument('--beam_size', default=2, type=int)
parser.add_argument('--num_tokens', default=20, type=int)
parser.add_argument('--top_p', default=0.9, type=float)
parser.add_argument('--model_size', default='medium', type=str)
parser.add_argument('--cache_dir', default=None, type=str)
parser.add_argument('--retriever_model_path', default='t5x_conversion/t5_xl_all', type=str)
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

t5x_embedder = RankGenEncoder(model_path=args.retriever_model_path, cache_dir=args.cache_dir)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(49)
random.shuffle(data)

random.seed(442)
random.shuffle(data)

folder_name = f"token_bs_t5x"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained(f"gpt2-{args.model_size}", cache_dir=args.cache_dir)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(f"gpt2-{args.model_size}", cache_dir=args.cache_dir)
model.to(device)
model.eval()


def postprocess(outputs):
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


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


def scorer_t5x(t5x_embedder, prefixes, suffixes, prefix_vectors=None):
    if prefix_vectors is None:
        prefix_vectors = t5x_embedder.encode(prefixes, vectors_type="prefix")["embeddings"]
    suffix_vectors = t5x_embedder.encode(suffixes, vectors_type="suffix")["embeddings"]
    matmul = torch.matmul(prefix_vectors, suffix_vectors.t()).squeeze(dim=0)
    similarities = torch.from_numpy(np.diagonal(matmul.cpu().numpy()))
    return similarities, prefix_vectors, suffix_vectors


def token_beam_search(contexts, scorer, beam_size=3, temperature=1.0, top_p=0.9, num_tokens=5, num_samples=10,
                      max_length=115):
    final_outputs = []
    final_scores = []
    total_generated_tokens = 0
    for ctx in contexts:
        ctx_tokens = tokenizer(ctx, truncation=True, padding="longest", return_tensors="pt")['input_ids']
        beams = [{
            "prefix_text": ctx,
            "prefix_tokens": ctx_tokens,
            "text": "",
            "eos": False
        } for _ in range(beam_size)]
        while True:
            all_outs = []
            max_new_tokens = min(num_tokens, max_length - total_generated_tokens)
            for beam in beams:
                # if a beam has ended, add it to all_outs
                if beam["eos"]:
                    all_outs.append(beam)
                    continue
                # otherwise generate the next n tokens
                inputs = beam['prefix_tokens'].to(device)
                num_input_tokens = inputs.size()[1]
                curr_outs = model.generate(inputs, do_sample=True, output_scores=True,
                                           return_dict_in_generate=True,
                                           max_new_tokens=max_new_tokens, top_k=None, top_p=top_p,
                                           num_return_sequences=num_samples, temperature=temperature)
                is_eos = []
                for curr_out in curr_outs['sequences']:
                    if tokenizer.eos_token_id in curr_out:
                        is_eos.append(True)
                    else:
                        is_eos.append(False)
                curr_outs_text = postprocess(curr_outs['sequences'][:, num_input_tokens:])
                for tokens, text, eos in zip(curr_outs['sequences'][:, num_input_tokens:], curr_outs_text, is_eos):
                    curr_outs_sents = sent_tokenize(text)
                    # if a full sentence has been generated
                    if len(curr_outs_sents) > 1:
                        # remove first sentence from prefix and append generated sentence to prefix
                        prefix_sents = nltk.sent_tokenize(beam["prefix_text"])[1:]
                        prefix_sents.append(curr_outs_sents[0])
                        while len(tokenizer(' '.join(prefix_sents))['input_ids']) > 256:
                            prefix_sents.pop(0)
                        prefix_text = ' '.join(prefix_sents)
                        prefix_tokens = tokenizer(prefix_text, truncation=True, padding="longest", return_tensors="pt")[
                            'input_ids']
                        prefix_tokens = prefix_tokens.to(device)
                        all_outs.append({
                            "prefix_text": prefix_text,
                            "prefix_tokens": prefix_tokens,
                            "text": beam["text"] + text,
                            "eos": eos
                        })
                    else:
                        all_outs.append({
                            "prefix_text": beam['prefix_text'],
                            "prefix_tokens": beam['prefix_tokens'],
                            "text": beam["text"] + text,
                            "eos": eos
                        })

            # Each beam has total_generated_tokens length
            total_generated_tokens += max_new_tokens
            if len(all_outs) > 1:
                # skip beam scoring if only one output to choose from
                scores, _, _ = scorer(prefixes=[x["prefix_text"] for x in all_outs], suffixes=[x["text"] for x in all_outs], prefix_vectors=None)
                top_scores, top_indices = torch.topk(scores, k=beam_size)
                beams = [all_outs[x] for x in top_indices]  # only track the top k beams
            else:
                top_scores = torch.Tensor([1.0])
                top_scores.cuda()
                beams = all_outs

            for beam in beams:
                if len(tokenizer.tokenize(beam["text"])) >= max_length:
                    beam["eos"] = True

            if all([x["eos"] for x in beams]):
                final_outputs.append([x["text"] for x in beams])
                final_scores.append(top_scores)
                break
    return final_outputs, final_scores


scorer_fn = partial(scorer_t5x, t5x_embedder=t5x_embedder)

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
    token_beam_text, token_beam_scores = token_beam_search(contexts=[instance["prefix"]], scorer=scorer_fn,
                                                           beam_size=args.beam_size,
                                                           top_p=args.top_p, num_tokens=args.num_tokens,
                                                           num_samples=args.num_samples)

    token_beam_text = token_beam_text[0]
    token_beam_text = [truncate(" ".join(x.split())) for x in token_beam_text]
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
