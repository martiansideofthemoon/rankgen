import argparse
import json
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from utils import extend_sequence
from transformers import AutoTokenizer

nltk.download('punkt')

parser = argparse.ArgumentParser()
parser.add_argument('--source', default="rankgen_vary_lens_splits/wikipedia_eval_512_128_sent_boundary.jsonl")
parser.add_argument('--reference', default="rankgen_vary_lens_splits/wikipedia_eval_64_128_sent_boundary.jsonl")
parser.add_argument('--output_path', default="rankgen_vary_lens_splits/wikipedia_eval_{prefix_length}_128_sent_boundary.jsonl")
parser.add_argument('--prefix_length', default=128, type=int)
args = parser.parse_args()

args.output_path = args.output_path.replace("{prefix_length}", str(args.prefix_length))

with open(args.source, "r") as f:
    data_source = [json.loads(x) for x in f.read().strip().split("\n")]

with open(args.reference, "r") as f:
    data_ref = [json.loads(x) for x in f.read().strip().split("\n")]
    data_ref_dict = {dd['targets'][0]: 1 for dd in data_ref}

data_src_filt = [dd for dd in data_source if dd['targets'][0] in data_ref_dict]
t5_tokenizer = AutoTokenizer.from_pretrained("t5-large")

avg_lens = []

for dd in data_src_filt:
    new_prefix_sents = []
    sents = sent_tokenize(dd['prefix'])
    sent_lens = [len(t5_tokenizer.tokenize(x)) for x in sents]
    output, _ = extend_sequence(sents, sent_lens, len(sents) - 1, args.prefix_length, False)
    dd['prefix'] = output
    avg_lens.append(len(t5_tokenizer.tokenize(output)))
print(np.mean(avg_lens))

with open(args.output_path, "w") as f:
    f.write("\n".join([json.dumps(x) for x in data_src_filt]) + "\n")
