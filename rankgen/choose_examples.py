import random
import os
import argparse
from utils import export_server
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--folder', default="rankgen_train_data_samples/pg19")
parser.add_argument('--output_dir', default="rankgen_train_data_samples/pg19_html")
args = parser.parse_args()

files = os.listdir(args.folder)

books = []
for file in tqdm.tqdm(files):
    with open(f"{args.folder}/{file}", 'r') as f:
        data = [x.split('\t') for x in f.read().strip().split("\n")]
    random.shuffle(data)
    data = data[:100]
    output = ""
    for dd in data:
        output += f"<b>PREFIX</> = {dd[0]}\n\n"
        output += f"<b>SUFFIX</> = {dd[1]}\n\n"
        output += f"<b>NEGATIVE</> = {dd[-1]}\n\n--------------------------\n\n"

    export_server(output, os.path.join(args.output_dir, file))

random.shuffle(books)
