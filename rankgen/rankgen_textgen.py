import torch
import argparse
import os
from rankgen import RankGenEncoder, RankGenGenerator


os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--rankgen_encoder', default='kalpeshk2011/rankgen-t5-xl-all', type=str)
parser.add_argument('--cache_dir', default=None, type=str)
args = parser.parse_args()

rankgen_encoder = RankGenEncoder(model_path=args.rankgen_encoder, cache_dir=args.cache_dir)
rankgen_generator = RankGenGenerator(rankgen_encoder=rankgen_encoder, language_model="gpt2-{args.model_size}", cache_dir=args.cache_dir)


def textgen(prefix, suffix, epochs):
    prefix_vector = rankgen_encoder.encode(prefix, vectors_type="prefix")["embeddings"]
    suffix_vector = rankgen_encoder.encode(suffix, vectors_type="suffix")["embeddings"]
    similarity = torch.matmul(prefix_vector, suffix_vector.t()).squeeze(dim=0)
    for i in range(epochs):
        print(f"Epoch {i}")
    return


pre = "pre "
suf = "suf You"
