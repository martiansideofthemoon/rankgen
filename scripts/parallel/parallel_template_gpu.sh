#!/bin/sh
#SBATCH --job-name=job_<exp_id>_<local_rank>
#SBATCH -o /mnt/nfs/work1/miyyer/kalpesh/projects/presuf-retrieval/scripts/parallel/parallel_logs/logs_exp_<exp_id>/log_<local_rank>.txt
#SBATCH --time=4:00:00
#SBATCH --partition=<gpu>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=45GB
#SBATCH -d singleton

cd /mnt/nfs/work1/miyyer/kalpesh/projects/presuf-retrieval

<command> --local_rank <local_rank> --num_shards <total>
