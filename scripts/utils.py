import math
import re
import string
import pickle
import collections as cll
import torch
import numpy as np
from scipy.stats import kendalltau


def cudafy_tokens(tokens):
    for x, y in tokens.items():
        tokens[x] = y.cuda()
    return tokens


def form_partitions(dataset, num_shards):
    p_indices = np.round(np.linspace(0, len(dataset), num_shards + 1))
    p_indices = [int(x) for x in p_indices]
    partitions = [dataset[p_indices[i]:p_indices[i + 1]] for i in range(len(p_indices) - 1)]
    assert len(partitions) == num_shards
    return partitions


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


def execute_gpt2(relevant_window, text_token_ids, tokenizer, model, output_hidden_states=False):
    num_ans_tokens = len(text_token_ids[0])
    inputs = tokenizer(" " + relevant_window, return_tensors="pt")
    inputs = cudafy_tokens(inputs)
    assert torch.equal(
        inputs["input_ids"][0, -1 * num_ans_tokens:],
        text_token_ids[0]
    )

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"], output_hidden_states=output_hidden_states)

        text_logits = outputs["logits"][0, -1 * num_ans_tokens - 1:-1, :]

        text_softmax = torch.nn.functional.softmax(text_logits, dim=1)
        softmax_ranks = torch.argsort(text_softmax, dim=1, descending=True)
        text_probs = torch.gather(text_softmax, 1, text_token_ids.t())
        log_probs = torch.log(text_probs)
        ppl = torch.exp(-1 * log_probs.sum() / num_ans_tokens).item()

        ranks = [softmax_ranks[i].tolist().index(text_token_ids[0][i].item()) + 1 for i in range(num_ans_tokens)]

    if output_hidden_states:
        return outputs["hidden_states"][-1][0, -1 * num_ans_tokens - 1:-1, :]
    else:
        return ranks, text_probs.squeeze().cpu().numpy(), log_probs.squeeze().cpu().numpy(), ppl

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth, gram=1, stopwords=None):
    """Calculate word level F1 score."""
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    prediction_tokens = [
        " ".join(prediction_tokens[i:i + gram])
        for i in range(0, len(prediction_tokens) - gram + 1)
    ]
    ground_truth_tokens = [
        " ".join(ground_truth_tokens[i:i + gram])
        for i in range(0, len(ground_truth_tokens) - gram + 1)
    ]

    if stopwords:
        prediction_tokens = [x for x in prediction_tokens if x not in stopwords]
        ground_truth_tokens = [x for x in ground_truth_tokens if x not in stopwords]

    if not prediction_tokens and not ground_truth_tokens:
        return 1.0, 1.0, 1.0
    common = cll.Counter(prediction_tokens) & cll.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def pickle_load(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def pickle_dump(file, data):
    with open(file, "wb") as f:
        pickle.dump(data, f)
