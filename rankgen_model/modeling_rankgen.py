import torch
import tqdm
from torch import nn
from transformers import T5PreTrainedModel, T5EncoderModel

class T5EncoderWithProjection(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.t5_encoder = T5EncoderModel(config)
        self.projection = nn.Linear(config.d_model, config.d_model, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, **input_args):
        hidden_states = self.t5_encoder(**input_args).last_hidden_state
        hidden_states = hidden_states[:, 0, :]
        batch_embeddings = self.projection(hidden_states)
        return batch_embeddings
