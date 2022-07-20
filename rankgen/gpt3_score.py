# Functions for receiving and evaluating GPT-3 response
import openai
import math
import numpy as np
import os
import json
import argparse
import random

from utils import pickle_load, pickle_dump

openai.api_key = os.environ['OPENAI_API_KEY']

def get_response(prompt: str, max_tokens = 150, temperature = 0.7, top_p = 1, n = 1, logprobs = 1, stop = None, echo = True):
    response = openai.Completion.create(engine="davinci",
                                        prompt=prompt,
                                        max_tokens=max_tokens,
                                        temperature = temperature,
                                        top_p=top_p,
                                        n=n,
                                        logprobs=logprobs,
                                        stop=stop,
                                        echo=echo)
    return response

def perplexity(log_probs):
    N = len(log_probs)
    return math.exp((-1/N) * np.sum(log_probs))

# Use max_tokens value passed to response to extract response PPL
def evaluate_response(response, max_tokens):
    response_dict = dict(response['choices'][0])
    text = response_dict['text']

    log_probs = response_dict['logprobs']['token_logprobs'][1:]
    log_probs_prompt = log_probs[:-max_tokens]
    log_probs_response = log_probs[-max_tokens:]

    ppl_prompt = perplexity(log_probs_prompt)
    ppl_response = perplexity(log_probs_response)
    ppl_total = perplexity(log_probs)

    return {
        'prompt_ppl': ppl_prompt,
        'response_ppl': ppl_response,
        'overall_ppl': ppl_total,
        'text': text,
    }

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="data/t5_xl_all_domains_wiki_hard.jsonl")
args = parser.parse_args()

with open(args.dataset, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]

gold_beats_neg_avg = []
gold_beats_neg_all = []
gold_beats_neg_any = []

for dd in data[:100]:
    if "gold_gpt3" in dd:
        print("skipping API call")
        write = False
    else:
        write = True
        gold = get_response(dd['prefix'].strip() + " " + dd['suffix'].strip(), 0)
        negs = []
        for nn in dd['negatives']:
            negs.append(
                get_response(dd['prefix'].strip() + " " + nn.strip(), 0)
            )
        dd['gold_gpt3'] = gold['choices'][0]['logprobs']
        dd['negs_gpt3'] = [nn['choices'][0]['logprobs'] for nn in negs]

    # prefix_len = 0
    # prefix_so_far = ""
    # for token in dd['gold_gpt3']['tokens']:
    #     prefix_so_far += token
    #     prefix_len += 1
    #     if prefix_so_far == dd['prefix']:
    #         break
    # if prefix_len > len(dd['gold_gpt3']['tokens']) - 10:
    #     continue
    # print(f"Suffix length = {len(dd['gold_gpt3']['tokens']) - prefix_len}")

    # gold_ppl = perplexity(dd['gold_gpt3']['token_logprobs'][prefix_len:])
    # neg_ppls = [perplexity(nn['token_logprobs'][prefix_len:]) for nn in dd['negs_gpt3']]

    gold_ppl = perplexity(dd['gold_gpt3']['token_logprobs'][1:])
    neg_ppls = [perplexity(nn['token_logprobs'][1:]) for nn in dd['negs_gpt3']]
    print(gold_ppl)
    print(neg_ppls)

    gold_beats_neg_avg.extend(
        [gold_ppl < nppl for nppl in neg_ppls]
    )
    gold_beats_neg_all.append(
        all([gold_ppl < nppl for nppl in neg_ppls])
    )
    gold_beats_neg_any.append(
        any([gold_ppl < nppl for nppl in neg_ppls])
    )

    print(f"Avg = {np.mean(gold_beats_neg_avg)} ({len(gold_beats_neg_avg)} instances)")
    print(f"All = {np.mean(gold_beats_neg_all)} ({len(gold_beats_neg_all)} instances)")
    print(f"Any = {np.mean(gold_beats_neg_any)} ({len(gold_beats_neg_any)} instances)")

    if write:
        output = "\n".join([json.dumps(x) for x in data]) + "\n"
        with open(args.dataset, "w") as f:
            f.write(output)
