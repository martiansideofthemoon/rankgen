import argparse
import os
import datetime
import time
import subprocess
import socket

# example to cancel jobs
# squeue -u $USER | grep "job_6" | awk '{print $1}' | tail -n +2 | xargs scancel


def get_run_id():
    filename = "rankgen/parallel/parallel_logs/expts.txt"
    if os.path.isfile(filename) is False:
        with open(filename, 'w') as f:
            f.write("")
        return 0
    else:
        with open(filename, 'r') as f:
            expts = f.readlines()
        run_id = len(expts) / 5
        print(len(expts))
    return run_id


parser = argparse.ArgumentParser()
parser.add_argument('--command', default="python rankgen/gpt2_generate.py --model_size medium --output_file outputs/wiki_gpt2_medium_typical_p90.tsv --num_samples 20 --typical_p 0.9")
parser.add_argument('--num_shards', default=20, type=int)
parser.add_argument('--start_shard', default=None, type=int)
parser.add_argument('--end_shard', default=None, type=int)
parser.add_argument('--partition_type', default="gypsum-2080ti", type=str)
args = parser.parse_args()

script_command = args.command
exp_id = int(get_run_id())
print(script_command)

TOTAL = args.num_shards
start_to_schedule = args.start_shard or 0
end_to_schedule = args.end_shard or args.num_shards

print(exp_id)
gpu_list = [args.partition_type for i in range(40)]

template = "rankgen/parallel/parallel_template_gpu.sh"

print(template)

with open(template, "r") as f:
    schedule_template = f.read()

for i in range(start_to_schedule, end_to_schedule):

    curr_gpu = gpu_list[i % len(gpu_list)]

    os.makedirs("rankgen/parallel/parallel_schedulers/schedulers_exp_%d" % exp_id, exist_ok=True)
    os.makedirs("rankgen/parallel/parallel_logs/logs_exp_%d" % exp_id, exist_ok=True)

    curr_template = schedule_template.replace("<total>", str(TOTAL)).replace("<local_rank>", str(i))
    curr_template = curr_template.replace("<exp_id>", str(exp_id)).replace("<command>", script_command)
    curr_template = curr_template.replace("<gpu>", curr_gpu)

    with open("rankgen/parallel/parallel_schedulers/schedulers_exp_%d/schedule_%d.sh" % (exp_id, i), "w") as f:
        f.write(curr_template + "\n")

    command = "sbatch rankgen/parallel/parallel_schedulers/schedulers_exp_%d/schedule_%d.sh" % (exp_id, i)
    print(subprocess.check_output(command, shell=True))
    time.sleep(0.2)

output = f"Experiment ID {exp_id}\n" + \
    "Script Command = " + script_command + "\n" + \
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "\n" + \
    "{:d} shards, {:d} - {:d} scheduled".format(TOTAL, start_to_schedule, end_to_schedule) + "\n" + \
    "" + "\n\n"

with open("rankgen/parallel/parallel_logs/expts.txt", "a") as f:
    f.write(output)
