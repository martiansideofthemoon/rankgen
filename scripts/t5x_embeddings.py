from transformers import T5Tokenizer, T5EncoderModel
import pickle
import argparse
import numpy as np
import tqdm
import os
import torch
import random
import json


class T5XEmbeddingGenerator():
    def __init__(self, max_batch_size=32, model_path='.', cache_dir=None):
        self.max_batch_size = max_batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        with open(os.path.join(model_path, 'state_dict.pickle'), 'rb') as handle:
            state_dict = pickle.load(handle)

        state_dict_new = {}
        for k, v in state_dict.items():
            if k != "encoder.embed_tokens.weight":
                v = np.transpose(v)
                state_dict_new[k] = torch.Tensor(v)
            else:
                state_dict_new[k] = torch.Tensor(v)
                state_dict_new["shared.weight"] = torch.Tensor(v)

        with open(os.path.join(model_path, 'projection.pickle'), 'rb') as handle:
            self.projection = torch.Tensor(pickle.load(handle))  # (1024, 1024), numpy array

        self.projection = self.projection.to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-large", cache_dir=cache_dir)
        model = T5EncoderModel.from_pretrained("google/t5-v1_1-large", cache_dir=cache_dir)
        self.model, missing_keys, unexpected_keys, mismatched_keys, error_msg = T5EncoderModel._load_state_dict_into_model(
            model,
            state_dict_new,
            "google/t5-v1_1-large")
        self.model.to(self.device)
        self.model.eval()

    def encode(self, inputs, vectors_type="prefix", verbose=False):
        tokenizer = self.tokenizer
        max_batch_size = self.max_batch_size
        if isinstance(inputs, str):
            inputs = [inputs]
        if vectors_type == 'prefix':
            inputs = ['pre ' + input for input in inputs]
            max_length = 512
        else:
            inputs = ['suffi ' + input for input in inputs]
            max_length = 128

        all_embeddings = []
        for i in tqdm.tqdm(range(0, len(inputs), max_batch_size), total=(len(inputs) // max_batch_size) + 1, disable=not verbose, desc=f"Encoding {vectors_type} inputs:"):
            tokenized_inputs = tokenizer(inputs[i:i + max_batch_size], return_tensors="pt", padding=True)
            for k, v in tokenized_inputs.items():
                tokenized_inputs[k] = v[:, :max_length]
            tokenized_inputs = tokenized_inputs.to(self.device)
            with torch.no_grad():
                hidden_states = self.model(**tokenized_inputs).last_hidden_state
                hidden_states = hidden_states[:, 0, :]
                batch_embeddings = torch.matmul(hidden_states, self.projection)
            all_embeddings.append(batch_embeddings)
        return {"embeddings": torch.cat(all_embeddings, dim=0)}
