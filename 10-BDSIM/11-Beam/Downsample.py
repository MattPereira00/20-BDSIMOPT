import random

input_file = "LhARA_0cm_pm2.dat"
output_file = "LhARA_0cm_pm2-100k.dat"
target_size = 100000

# First count total lines (particles)
with open(input_file, "r") as f:
    total_lines = sum(1 for _ in f)

# Randomly select which line indices to keep
sample_indices = set(random.sample(range(total_lines), target_size))

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for i, line in enumerate(fin):
        if i in sample_indices:
            fout.write(line)
