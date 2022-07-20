## RankGen - Improving Text Generation with Large Ranking Models

This is the official repository for our preprint, [RankGen - Improving Text Generation with Large Ranking Models](https://arxiv.org/abs/2205.09726). RankGen is a 1.2 billion encoder model which maps prefixes and generations from any pretrained English language model to a shared vector space. RankGen can be used to rerank multiple full-length samples from an LM, and it can also be incorporated as a scoring function into beam search to significantly improve generation quality (0.85 vs 0.77 [MAUVE](https://arxiv.org/abs/2102.01454), 75% preference according to humans annotators who are English writers).

This repository contains human evaluation data, links to HuggingFace-compatible model checkpoints, and code to integrate RankGen in beam search on HuggingFace models. RankGen is trained by fine-tuning the T5-XL encoder using the [T5X library](https://github.com/google-research/t5x).

### Updates

* (July 2022) RankGen checkpoints are now available on the HuggingFace Model Hub ([link](https://huggingface.co/kalpeshk2011))!

### Model checkpoints

All RankGen checkpoints are available on the HuggingFace Model Hub - [link](https://huggingface.co/kalpeshk2011)

We recommend using `RankGen-XL-all`.

| Checkpoint        | Size | Model Name                        | HF Hub Link                                                      |
|-------------------|------|-----------------------------------|------------------------------------------------------------------|
| RankGen-base-all  | 0.1B | kalpeshk2011/rankgen-t5-base-all  | [link](https://huggingface.co/kalpeshk2011/rankgen-t5-base-all)  |
| RankGen-large-all | 0.3B | kalpeshk2011/rankgen-t5-large-all | [link](https://huggingface.co/kalpeshk2011/rankgen-t5-large-all) |
| RankGen-XL-all    | 1.2B | kalpeshk2011/rankgen-t5-xl-all    | [link](https://huggingface.co/kalpeshk2011/rankgen-t5-xl-all)    |
| RankGen-XL-PG19   | 1.2B | kalpeshk2011/rankgen-t5-xl-pg19   | [link](https://huggingface.co/kalpeshk2011/rankgen-t5-xl-pg19)   |

*Older versions of the checkpoints*:

RankGen XL checkpoints compatible with `T5XEmbeddingGeneratorLegacy` - [here](https://drive.google.com/drive/folders/1m8ujkAqkBBWYAJISZigz1Lw4tQGbZXaY?usp=sharing)

T5X JAX checkpoints (base, large, XL) - [here](https://github.com/google-research/google-research/tree/master/rankgen)

### Setup

**Installation**

```
virtualenv rankgen-venv
source rankgen-venv/bin/activate
pip install torch torchvision # currently, this is the version compatible with CUDA 10.1
pip install transformers
pip install sentencepiece
pip install gdown # optional dependency
```

**Data Download**

Get the data [here](https://drive.google.com/drive/folders/1DRG2ess7fK3apfB-6KoHb_azMuHbsIv4?usp=sharing) and place folder in root directory. Alternatively, use `gdown` as shown below,

```
gdown --folder https://drive.google.com/drive/folders/1DRG2ess7fK3apfB-6KoHb_azMuHbsIv4
```

Run the test script to make sure the RankGen checkpoint has loaded correctly,

```
python scripts/test_t5x_embeddings.py --model_path kalpeshk2011/rankgen-t5-base-all

### Expected output
0.0009239262409127233
0.0011521980725477804
```

### Using RankGen

Loading RankGen is simple using the HuggingFace APIs, but we suggest using [`T5XEmbeddingGenerator`](scripts/t5x_embeddings.py) for correctly processing data.

```
from transformers import T5Tokenizer, AutoModel

self.tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xl", cache_dir=cache_dir)
self.model = AutoModel.from_pretrained("kalpeshk2011/rankgen-t5-xl-all", trust_remote_code=True)
```

In order to perform tokenization correctly & simplify the data preprocessing code, we have wrapped this into the [`T5XEmbeddingGenerator`](scripts/t5x_embeddings.py) class. Please see [`scripts/test_t5x_embeddings.py`](scripts/test_t5x_embeddings.py) for an example of the usage (below is a snippet from the file).

```
from t5x_embeddings import T5XEmbeddingGenerator
hf_model = T5XEmbeddingGenerator(args.model_path)

with torch.inference_mode():
    all_prefix_outs = hf_model.encode([x["inputs"]["inputs_pretokenized"] for x in examples], vectors_type="prefix", verbose=True, return_input_ids=True)
```

### Running beam search with RankGen

The main file is [`scripts/rankgen_beam_search.py`](scripts/rankgen_beam_search.py). To execute it,

```
python scripts/rankgen_beam_search.py \
    --dataset rankgen_data/wiki.jsonl \
    --retriever_model_path kalpeshk2011/rankgen-t5-xl-all \
    --num_tokens 20 --num_samples 10 --beam_size 2 \
    --output_file outputs_beam/wiki_t5_xl_beam_2_tokens_20_samples_10.jsonl
```

Evaluating using MAUVE (make sure JSONL file has several thousand generations for intuitive MAUVE scores, 7713 in our experiments),

```
python scripts/score_multi_beam.py --dataset outputs_beam/wiki_t5_xl_beam_2_tokens_10_samples_10.jsonl
```


### Human evaluation data

We conducted our human evaluation on Upwork, hiring English teachers and writers. We performed blind A/B testing between RankGen and nucleus sampling. We also asked our annotators to provide a 1-3 sentence explanation. You can find all the 600 annotations across two files in [`human-eval-data`](human-eval-data). To compute the evaluation scores run,

```
python scripts/score_ab_text.py
```

### Citation Information
If you use RankGen, please cite it as follows:
```
@article{krishna2022rankgen,
  title={RankGen: Improving Text Generation with Large Ranking Models},
  author={Kalpesh Krishna and Yapei Chang and John Wieting and Mohit Iyyer},
  journal={arXiv preprint arXiv:2205.09726},
  year={2022}
}
```
