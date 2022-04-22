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
    def __init__(self, max_batch_size=32, model_path='.', model_size=None, cache_dir=None):
        self.max_batch_size = max_batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_size is None:
            if "t5_large" in model_path:
                self.model_size = "large"
            elif "t5_xl" in model_path:
                self.model_size = "xl"
            else:
                self.model_size = "base"
        else:
            self.model_size = model_size

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
        self.tokenizer = T5Tokenizer.from_pretrained(f"google/t5-v1_1-{self.model_size}", cache_dir=cache_dir)
        model = T5EncoderModel.from_pretrained(f"google/t5-v1_1-{self.model_size}", cache_dir=cache_dir)
        self.model, _, _, _, _ = T5EncoderModel._load_pretrained_model(model, state_dict_new, None, f"google/t5-v1_1-{self.model_size}")
        self.model.to(self.device)
        self.model.eval()

    def encode(self, inputs, vectors_type="prefix", verbose=False, return_input_ids=False):
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
        all_input_ids = []
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
            if return_input_ids:
                all_input_ids.extend(tokenized_inputs.input_ids.cpu().tolist())
        return {
            "embeddings": torch.cat(all_embeddings, dim=0),
            "input_ids": all_input_ids
        }
