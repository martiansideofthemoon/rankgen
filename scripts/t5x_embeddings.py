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
from utils import export_server, clean_token
import nltk

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class T5XEmbeddingGenerator():
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

        with open('state_dict.pickle', 'rb') as handle:
            state_dict = pickle.load(handle)

        state_dict_new = {}
        for k, v in state_dict.items():
            if k != "encoder.embed_tokens.weight":
                v = np.transpose(v)
                state_dict_new[k] = torch.Tensor(v)
            else:
                state_dict_new[k] = torch.Tensor(v)
                state_dict_new["shared.weight"] = torch.Tensor(v)

        with open('projection.pickle', 'rb') as handle:
            self.projection = torch.Tensor(pickle.load(handle))  # (1024, 1024), numpy array

        self.projection = self.projection.to(device)
        self.tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-large")
        model = T5EncoderModel.from_pretrained("google/t5-v1_1-large")
        self.model, missing_keys, unexpected_keys, mismatched_keys, error_msg = T5EncoderModel._load_state_dict_into_model(
            model,
            state_dict_new,
            "google/t5-v1_1-large")
        self.model.to(device)

    def create_mini_batches(self, all_encodings):
        batch_size = self.batch_size
        all_batch_input_ids = []
        all_batch_attention_masks = []
        n_encodings = all_encodings.input_ids.size()[0]
        n_minibatches = n_encodings // batch_size
        i = 0
        for i in range(n_minibatches):
            batch_input_ids = all_encodings.input_ids[i * batch_size:(i + 1) * batch_size]
            all_batch_input_ids.append(batch_input_ids)
            batch_attention_masks = all_encodings.attention_mask[i * batch_size:(i + 1) * batch_size]
            all_batch_attention_masks.append(batch_attention_masks)
        if n_encodings % batch_size != 0:
            batch_input_ids = all_encodings.input_ids[i * batch_size:n_encodings]
            all_batch_input_ids.append(batch_input_ids)
            batch_attention_masks = all_encodings.attention_mask[i * batch_size:n_encodings]
            all_batch_attention_masks.append(batch_attention_masks)
        return all_batch_input_ids, all_batch_attention_masks

    def encode(self, inputs, vectors_type="prefix"):
        embeddings = torch.zeros([1, 1])
        tokenizer = self.tokenizer
        if vectors_type == 'prefix':
            inputs = ['pre ' + input for input in inputs]
            all_encodings = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
        else:
            inputs = ['suffi ' + input for input in inputs]
            all_encodings = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=128)
        all_encodings = all_encodings.to(device)
        all_batch_input_ids, all_batch_attention_masks = self.create_mini_batches(all_encodings)
        for batch_input_ids, batch_attention_masks in zip(all_batch_input_ids, all_batch_attention_masks):
            hidden_states = self.model(input_ids=batch_input_ids,
                                       attention_mask=batch_attention_masks).last_hidden_state  # (batch_size, 512, 1024), tensor
            proj = self.projection.repeat(hidden_states.size()[0], 1, 1)
            projections = torch.matmul(hidden_states, proj) # (batch_size, 512, 1024) by (batch_size, 1024, 1024)
            batch_embeddings = projections[:, 0, :]  # (batch_size, 1024), tensor
            if embeddings.size() == (1, 1):
                embeddings = batch_embeddings
            else:
                embeddings = torch.cat((embeddings, batch_embeddings), 0)
        return embeddings


nltk.download('punkt')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="/mnt/nfs/work1/miyyer/kalpesh/projects/presuf-retrieval/data/t5_xl_all_domains_pg19_random.jsonl", type=str)
parser.add_argument('--compare_between', default="GLD,PPL,RET", type=str)
parser.add_argument('--num_samples', default=40, type=int)
parser.add_argument('--beam_size', default=1, type=int)
parser.add_argument('--generation_length', default=None, type=int)
parser.add_argument('--num_tokens', default=5, type=int)
parser.add_argument('--top_p', default=0.9, type=float)
parser.add_argument('--export', dest='export', action='store_true')
parser.add_argument('--model_size', default='medium', type=str)
args = parser.parse_args()

with open(args.dataset, "r") as f:
    data = [json.loads(x) for x in f.read().strip().split("\n")]
t5x = T5XEmbeddingGenerator(batch_size=5)

random.seed(49)
random.shuffle(data)

num_samples = args.num_samples
orig_prec_scores = []
orig_f1_scores = []
baseline_prec_scores = []
baseline_f1_scores = []
generated_prec_scores = []
generated_f1_scores = []

def scorer_t5x(t5x, prefix, suffixes, prefix_vector=None):
    #
    if prefix_vector is None:
        prefix_vector = t5x.encode([prefix], vectors_type="prefix")
    suffix_vectors = t5x.encode(suffixes, vectors_type="suffix")
    print(prefix_vector.size())
    print(suffix_vectors.size())
    similarities = torch.matmul(prefix_vector, suffix_vectors.t()).squeeze()
    return similarities, prefix_vector, suffix_vectors


exact_matches = 0
random.seed(442)
random.shuffle(data)

folder_name = f"token_bs_t5x"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained(f"gpt2-{args.model_size}")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(f"gpt2-{args.model_size}")
model.cuda()
model.eval()

def postprocess(outputs):
    return "".join(tokenizer.batch_decode(outputs, skip_special_tokens=True))

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

def token_beam_search(contexts, scorer, beam_size=3, temperature=1.0, top_p=0.9, num_tokens=5, num_samples=40):
    final_outputs = []
    final_scores = []
    for ctx in contexts:
        _, prefix_vector, _ = scorer(prefix=ctx, suffixes=[ctx])
        num_prefix_chars = len(ctx)
        num_prefix_tokens = len(tokenizer(ctx, truncation=True, padding="longest", return_tensors="pt")['input_ids'][0])
        beams = [{
            "text": ctx,
            "eos": False
        } for _ in range(beam_size)]
        while True:
            all_outs = []
            for beam in beams:
                # if a beam has ended, add it to all_outs
                if beam["eos"]:
                    all_outs.append(beam)
                # otherwise generate the next n tokens
                else:
                    inputs = tokenizer([beam['text'] for _ in range(num_samples)], truncation=True, padding="longest", return_tensors="pt").to(
                        device)
                    num_input_tokens = len(inputs['input_ids'][0])
                    curr_outs = model.generate(**inputs, do_sample=False, output_scores=True,
                                                       return_dict_in_generate=True,
                                                       max_new_tokens=num_tokens, top_p=top_p, temperature=temperature)
                    is_eos = []
                    for curr_out in curr_outs['sequences']:
                        if tokenizer.eos_token_id in curr_out:
                            is_eos.append(True)
                        else:
                            is_eos.append(False)
                    curr_outs = postprocess(curr_outs['sequences'][0, num_input_tokens:])
                    curr_outs = " ".join(curr_outs.split())
                    for text, eos in zip(curr_outs, is_eos):
                        # update all_outs
                        all_outs.append({
                            "text": beam["text"] + text,
                            "eos": eos
                        })
            scores, _, _ = scorer(prefix=ctx, suffixes=[x["text"][num_prefix_chars:] for x in all_outs], prefix_vector=prefix_vector)
            top_scores, top_indices = torch.topk(scores, k=beam_size)
            beams = [all_outs[x] for x in top_indices] # only track the top k beams

            for beam in beams:
                if len(tokenizer.tokenize(beam["text"])) >= 128:
                    beam["eos"] = True

            if all([x["eos"] for x in beams]):
                final_outputs.append(beams[0]["text"][num_prefix_chars:])
                final_scores.append(top_scores[0])
                break
    return final_outputs, final_scores

for kk, instance in tqdm.tqdm(enumerate(data), total=len(data)):
    if kk < 50:
        continue
    html_output = f"<b>Prefix</> = {instance['prefix']}\n\n"
    token_bs_output = token_beam_search(
        contexts=[instance["prefix"]], scorer=partial(scorer_t5x, t5x=t5x), beam_size=args.beam_size,
        temperature=1.0, top_p=args.top_p, num_tokens=args.num_tokens)[0][0]
    batch = tokenizer(instance['prefix'], truncation=True, padding="longest", return_tensors="pt").to(device)
    num_input_tokens = len(batch['input_ids'][0])
    generation_output = model.generate(**batch, do_sample=True, output_scores=True, return_dict_in_generate=True,
                                max_new_tokens=args.generation_length, top_p=args.top_p, temperature=1.0)
    generation_output = postprocess(generation_output['sequences'][0, num_input_tokens:])
    generation_output = " ".join(generation_output.split())
    generation_output = truncate(generation_output)
    print(generation_output)

    # TODO: consider "!" and "?"
    if "." not in token_bs_output or "." not in generation_output:
        continue

    suffixes = {
        "GLD-OUTP": instance["suffix"],
        "PPL-TOPP": generation_output,
        "RET-TOKN": token_bs_output
    }
    cmpr = ["PPL-TOPP", "RET-TOKN"]

    suffixes = [f"\033[0;37;47m {cmpr[0]} " + f"#{kk + 1}</> = {suffixes[cmpr[0]]}\n\n",
                f"\033[0;37;47m {cmpr[1]} " + f"#{kk + 1}</> = {suffixes[cmpr[1]]}\n\n"]

    if random.random() > 0.5:
        html_output += suffixes[0] + suffixes[1]
    else:
        html_output += suffixes[1] + suffixes[0]

    export_server(html_output, f"p{int(args.top_p * 100)}_eval_{kk}", folder_name)