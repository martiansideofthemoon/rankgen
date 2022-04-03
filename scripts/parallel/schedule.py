import os
import datetime
import time
import subprocess
import socket


def get_run_id():
    filename = "scripts/parallel_logs/expts.txt"
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


script_command = "python scripts/gpt2_generate.py --model_size medium --output_file outputs/wiki_gpt2_medium_typical_p90.tsv --num_samples 20 --typical_p 0.9"
exp_id = int(get_run_id())
print(script_command)
TOTAL = 40
start_to_schedule = 0
end_to_schedule = 40

print(exp_id)
gpu_list = ["2080ti-short" for i in range(30)] + ["1080ti-short" for i in range(30)] + 
# gpu_list = ["rtx8000-short" for i in range(50)]

template = "scripts/parallel_template_gpu.sh"

print(template)

with open(template, "r") as f:
    schedule_template = f.read()

for i in range(start_to_schedule, end_to_schedule):

    curr_gpu = gpu_list[i % len(gpu_list)]

    os.makedirs("scripts/parallel_schedulers/schedulers_exp_%d" % exp_id, exist_ok=True)
    os.makedirs("scripts/parallel_logs/logs_exp_%d" % exp_id, exist_ok=True)

    curr_template = schedule_template.replace("<total>", str(TOTAL)).replace("<local_rank>", str(i))
    curr_template = curr_template.replace("<exp_id>", str(exp_id)).replace("<command>", script_command)
    curr_template = curr_template.replace("<gpu>", curr_gpu)

    with open("scripts/parallel_schedulers/schedulers_exp_%d/schedule_%d.sh" % (exp_id, i), "w") as f:
        f.write(curr_template + "\n")

    command = "sbatch scripts/parallel_schedulers/schedulers_exp_%d/schedule_%d.sh" % (exp_id, i)
    print(subprocess.check_output(command, shell=True))
    time.sleep(0.2)

output = "Script Command = " + script_command + "\n" + \
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "\n" + \
    "{:d} shards, {:d} - {:d} scheduled".format(TOTAL, start_to_schedule, end_to_schedule) + "\n" + \
    "" + "\n\n"

with open("scripts/parallel/parallel_logs/expts.txt", "a") as f:
    f.write(output)
