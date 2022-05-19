## RankGen - Improving Text Generation with Large Ranking Models

This is the official repository for our preprint, "RankGen - Improving Text Generation with Large Ranking Models". RankGen is a 1.2 billion encoder model which maps prefixes and generations from any language model (in continutation to the prefix) to a shared vector space. Re-ranking with RankGen as well as integrating RankGen in beam search significantly improves generation quality (0.85 vs 0.77 [MAUVE](https://arxiv.org/abs/2102.01454), 75% preference according to humans annotators who are English writers).

This repository will contain human evaluation data, link to HuggingFace compatible model checkpoints, and code to integrate RankGen in beam search on HuggingFace models. RankGen is trained by fine-tuning the T5-XL encoder using the [T5X library](https://github.com/google-research/t5x).

### Paper

See [`paper.pdf`](paper.pdf) for now, it should be up on arXiv soon!

### Human evaluation data

We conducted our human evaluation on Upwork, hiring English teachers and writers. We performed blind A/B testing between RankGen and nucleus sampling. We also asked our annotators to provide a 1-3 sentence explanation. You can find all the 600 annotations across two files in [`human-eval-data`](human-eval-data). To compute the evaluation scores run,

```
python scripts/score_ab_text.py
```

### Model checkpoints

coming soon! (aiming for 26th May, 2022)

### Running beam search with RankGen

The main file is [`scripts/rankgen_beam_search.py`](scripts/rankgen_beam_search.py). This file will require RankGen checkpoints, which will be added to the repository soon!

**Setup** ---

```
virtualenv rankgen-venv
source rankgen-venv/bin/activate
pip install torch torchvision # currently, this is the version compatible with CUDA 10.1
pip install transformers
```

Run the test script to make sure the RankGen checkpoint has loaded correctly,

```
python scripts/test_t5x_embeddings.py --model_path t5x_conversion/t5_xl_all

### Expected output
0.0006388302952054898
0.0007493323556995418
```

Running beam search,

```
python scripts/rankgen_beam_search.py --retriever_model_path t5x_conversion/t5_xl_all \
    --num_tokens 20 --num_samples 10 --beam_size 2 --output_file outputs_beam/wiki_t5_xl_beam_2_tokens_20_samples_10.jsonl
```

Evaluating using MAUVE (make sure JSONL file has several thousand generations for intuitive MAUVE scores, 7713 in our experiments),

```
python scripts/score_multi_beam.py --dataset outputs_beam/wiki_t5_xl_beam_2_tokens_10_samples_10.jsonl
```
