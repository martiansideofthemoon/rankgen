import argparse
import glob
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--input_pattern', default="openwebtext_vectors/2016-06.pkl_0_small.pkl.matches_entity_*", type=str)
parser.add_argument('--output_file', default=None, type=str)
args = parser.parse_args()

files = glob.glob(args.input_pattern)
file_with_ids = [(int(f.split("_")[-1]), f) for f in files]
file_with_ids.sort(key=lambda x: x[0])

data = ""
for file in file_with_ids:
    with open(file[1], "r") as f:
        data += f.read()

if args.output_file is not None:
    output_file = args.output_file
else:
    output_file = ".".join(args.input_pattern.split(".")[:-1])
print(output_file)
with open(output_file, "w") as f:
    f.write(data)
