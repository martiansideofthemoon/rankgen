from transformers import T5Tokenizer, T5EncoderModel
import pickle
import numpy as np
import json
import torch
import os
import argparse

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class T5EmbeddingGenerator():
    def __init__(self):
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

        self.tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-large")
        model = T5EncoderModel.from_pretrained("google/t5-v1_1-large")
        self.model, missing_keys, unexpected_keys, mismatched_keys, error_msg = T5EncoderModel._load_state_dict_into_model(
            model,
            state_dict_new,
            "google/t5-v1_1-large")

    def preprocess_sentences(self, inputs, vectors_type="prefix"):
        tokenizer = self.tokenizer

        all_input_ids = []
        for input in inputs:
            if vectors_type == 'prefix':
                input = '[PRE]' + input
            else:
                input = '[SUF]' + input
            input_ids = tokenizer.encode(input)
            for i in range(len(input_ids), 512):
                input_ids.append(0)
            input_ids = torch.LongTensor(input_ids)
            input_ids = torch.unsqueeze(input_ids, 0)
            all_input_ids.append(input_ids)

        return all_input_ids

    def get_attention_masks(self, all_input_ids):
        all_attention_masks = []
        for input_ids in all_input_ids:
            attention_mask = (input_ids > 0).type(torch.int32)
            all_attention_masks.append(attention_mask)
        return all_attention_masks

    def create_mini_batches(self, all_input_ids, all_attention_masks, batch_size):
        all_batch_ids = []
        all_batch_attention_masks = []
        n_minibatches = len(all_input_ids) // batch_size
        i = 0
        for i in range(n_minibatches):
            batch_ids = all_input_ids[i * batch_size:(i + 1) * batch_size]
            all_batch_ids.append(batch_ids)
            batch_attention_masks = all_attention_masks[i * batch_size:(i + 1) * batch_size]
            all_batch_attention_masks.append(batch_attention_masks)
        if len(all_input_ids) % batch_size != 0:
            batch_ids = all_input_ids[i * batch_size:len(all_input_ids)]
            all_batch_ids.append(batch_ids)
            batch_attention_masks = all_attention_masks[i * batch_size:len(all_input_ids)]
            all_batch_attention_masks.append(batch_attention_masks)
        return all_batch_ids, all_batch_attention_masks

    def encode(self, all_input_ids, all_attention_masks, batch_size=5):
        embeddings = []
        all_batch_ids, all_batch_attention_masks = self.create_mini_batches(all_input_ids, all_attention_masks, batch_size)
        for batch_ids, batch_attention_masks in zip(all_batch_ids, all_batch_attention_masks):
            batch_ids = torch.stack(batch_ids)
            batch_attention_masks = torch.stack(batch_attention_masks)
            hidden_states = self.model(input_ids=batch_ids,
                                 attention_mask=batch_attention_masks).last_hidden_state  # (batch_size, 512, 1024), tensor
            self.projection = self.projection.repeat(batch_size, 1, 1)
            projections = torch.matmul(hidden_states, self.projection)
            batch_embeddings = projections[:, 0, :] # (batch_size, 1024), tensor
            embeddings.extend(batch_embeddings.tolist())
        return embeddings


generator = T5EmbeddingGenerator()
f = open(
        "t5x/pre_suf_retriever/checkpoint_1100000/pg19_valid_gold_beats_neg/pg19_debug_inference-score.jsonl-00000-of-00001.processed",
        "r")
examples = f.readlines()[0:5] # 5 is the batch size
examples = [json.loads(example) for example in examples]
prefix_ids = [torch.LongTensor(example['inputs_processed']['prefix_ids']) for example in examples]
input_embeddings = [example['score']['input_embedding'] for example in examples]
attention_masks = generator.get_attention_masks(prefix_ids)
embeddings = generator.encode(prefix_ids, attention_masks)
for i, embedding in enumerate(embeddings):
    diff = torch.Tensor(input_embeddings[i]) - torch.Tensor(embedding)
    print(diff)