import math
import re
import string
import pickle
import collections as cll
import torch
import numpy as np
import subprocess


def cudafy_tokens(tokens):
    for x, y in tokens.items():
        tokens[x] = y.cuda()
    return tokens


def extend_sequence(sents, sent_lens, start, limit, exceed_len=False,
                    direction='prefix', skip_sentences=None):
  """Extend a sequence by adding more sentences in prefix or suffix."""
  curr_value = start
  total_length = sent_lens[curr_value]
  full_sequence = sents[curr_value]
  assert len(sents) == len(sent_lens)

  if direction == 'prefix':
    increment = -1
    concat_fn = lambda curr, extra: extra + ' ' + curr
    continue_fn = lambda x: x >= 0
  else:
    increment = 1
    concat_fn = lambda curr, extra: curr + ' ' + extra
    continue_fn = lambda x: x < len(sents)

  while total_length < limit and continue_fn(curr_value + increment):
    proposed_length = total_length + sent_lens[curr_value + increment]
    if not exceed_len and proposed_length > limit:
      break
    if skip_sentences and (curr_value + increment) in skip_sentences:
      break
    curr_value += increment
    full_sequence = concat_fn(curr=full_sequence, extra=sents[curr_value])
    total_length += sent_lens[curr_value]

  if direction == 'prefix':
    assert curr_value <= start
  else:
    assert curr_value >= start

  return full_sequence, curr_value


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def postprocess(cls, input_str):
        input_str = input_str.replace("<h>", cls.HEADER)
        input_str = input_str.replace("<blue>", cls.OKBLUE)
        input_str = input_str.replace("<green>", cls.OKGREEN)
        input_str = input_str.replace("<yellow>", cls.WARNING)
        input_str = input_str.replace("<red>", cls.FAIL)
        input_str = input_str.replace("</>", cls.ENDC)
        input_str = input_str.replace("<b>", cls.BOLD)
        input_str = input_str.replace("<u>", cls.UNDERLINE)
        input_str = input_str.replace("<clean>", "")
        return input_str


def export_server(output, filename):
    with open("{}.txt".format(filename), "w") as f:
        f.write(Bcolors.postprocess(output) + "\n")
    subprocess.check_output("cat {0}.txt | ansi2html.sh --palette=linux --bg=dark > {0}.html".format(filename), shell=True)
    subprocess.check_output("rm {}.txt".format(filename), shell=True)


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
    if ";" in text:
        last_punc = max(last_punc, text.rindex(";"))
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


def rep_statistic(prefix, suffix, window=20):
    prefix_tokens = normalize_answer(prefix).split()
    suffix_tokens = normalize_answer(suffix).split()
    start_pos = len(prefix_tokens)
    tokens = prefix_tokens + suffix_tokens
    reps = [tokens[i] in tokens[i - window:i] for i in range(start_pos, len(tokens))]
    if len(reps) == 0:
        return 0.0
    else:
        return np.mean(reps)


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
