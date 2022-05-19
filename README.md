## RankGen

## Running token beam search

### Setup

0. Clone the repository https://github.com/martiansideofthemoon/presuf-retrieval

1. Make sure you have the latest HuggingFace `transformers` library.

2. Run the test script

```
python scripts/test_t5x_embeddings.py
```

2. The t5x checkpoint is located in `/mnt/nfs/work1/miyyer/kalpesh/projects/presuf-retrieval/t5x_conversion`

3. Running the script

```
cd /mnt/nfs/work1/miyyer/kalpesh/projects/presuf-retrieval

python scripts/parallel/schedule.py --command "python scripts/rankgen_beam_search.py --retriever_model_path t5x_conversion --num_tokens 10 --num_samples 10 --beam_size 2 --output_file outputs_beam/wiki_t5_large_beam_2_tokens_10_samples_10.jsonl" --num_shards 20 --partition_type "1080ti-short"

# See all expts in scripts/parallel/parallel_logs/expts.txt

cat scripts/parallel/parallel_logs/logs_exp_<expid>/log_<shard_id>.txt

grep -H Error scripts/parallel/parallel_logs/logs_exp_13/*
grep -H CANCEL scripts/parallel/parallel_logs/logs_exp_13/*

# If num_shards > 1 then run the command below
python scripts/parallel/merge.py --input_pattern "outputs_beam/wiki_t5_large_beam_2_tokens_10_samples_10.jsonl.shard*"

# Check file is right size (7713)
wc -l python scripts/score_multi_beam.py --dataset outputs_beam/wiki_t5_large_beam_2_tokens_10_samples_10.jsonl

# MAUVE score
python scripts/score_multi_beam.py --dataset outputs_beam/wiki_t5_large_beam_2_tokens_10_samples_10.jsonl
```

**Running without parallelization**

Takes longer (probably overnight), needs long queues, but less issues with sharding / Gypsum file system failures

* Find a `rtx8000-long` node, ask me if you need to run more. For low memory jobs you can also use `2080ti-long` or `1080ti-long`

**Running with parallelization**

* Found `1080ti-long` to be best. `rtx8000-long` / `rtx8000-short` seem to be reliable too.. Avoid `titanx` / `m40`. `2080ti-long` is a mixed bag