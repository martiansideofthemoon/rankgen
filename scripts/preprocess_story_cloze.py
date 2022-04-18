import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="data/story_cloze/story_cloze_spring_2016_test.csv")
args = parser.parse_args()

data = []

with open(args.dataset, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)

header = data[0]
data = data[1:]

output = ""
for dd in data:
    prefix = " ".join(dd[1:5])
    if dd[-1] == '1':
        output += f"{prefix}\t{dd[5]}\tplaceholder\tplaceholder\n"
        output += f"{prefix}\t{dd[6]}\tplaceholder\tplaceholder\n"
    elif dd[-1] == '2':
        output += f"{prefix}\t{dd[6]}\tplaceholder\tplaceholder\n"
        output += f"{prefix}\t{dd[5]}\tplaceholder\tplaceholder\n"
    else:
        raise ValueError("Wrong Answer Ending")

with open('data/story_cloze/story_cloze_spring_2016_test.tsv', 'w') as f:
    f.write(output)
