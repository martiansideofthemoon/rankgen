#!/bin/sh
#SBATCH --job-name=job_<exp_id>_<local_rank>
#SBATCH -o /work/kalpeshkrish_umass_edu/rankgen/rankgen/parallel/parallel_logs/logs_exp_<exp_id>/log_<local_rank>.txt
#SBATCH --partition=<gpu>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=45000
#SBATCH -d singleton

cd /work/kalpeshkrish_umass_edu/rankgen

<command> --local_rank <local_rank> --num_shards <total>
