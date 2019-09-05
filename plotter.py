import csv
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

i = 0


def extract(name):
    with open('reports.csv') as fd:
        reader = csv.reader(fd)
        for row in reader:
            if name in row[0]:
                return row


for s, d, f in os.walk('reward_plots'):

    for n in tqdm(f):
        if 'attacker' in n:
            with open(f'{s}/{n}', 'r') as fd:
                reader = csv.reader(fd)
                next(reader)

                data = []

                for row in reader:
                    data.append((int(row[1]), float(row[2])))

                plt.plot(
                    [x[0] for x in data],
                    [y[1] for y in data]
                )
    break

# plt.savefig('plot.png', dpi=850)
plt.show()