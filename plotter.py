import csv
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 9))


def extract(name):
    with open('reports.csv') as fd:
        reader = csv.reader(fd)
        for row in reader:
            if name in row[0]:
                return row

plot_data = {
    'rewards': {'x': [], 'y': []},
    'eps': {'x': [], 'y': []},
    'action': {'x': [], 'y': []},
    'loss': {'x': [], 'y': []},
    'td_error': {'x': [], 'y': []}
}


for subdir, d, f in os.walk('reward_plots'):
    for n in tqdm(f):
        with open(f'{subdir}/{n}', 'r') as fd:
            reader = csv.reader(fd)
            next(reader)

            data = []

            for row in reader:
                data.append([float(c) for c in row])

            s = sorted(zip(
                [x[1] for x in data],
                [y[2] for y in data]))
            plot_data['rewards']['x'].append([x[0] for x in s])
            plot_data['rewards']['y'].append([y[1] for y in s])

            s = sorted(zip(
                [x[4] for x in data],
                [y[5] for y in data]))
            plot_data['eps']['x'].append([x[0] for x in s])
            plot_data['eps']['y'].append([y[1] for y in s])

            s = sorted(zip(
                [x[7] for x in data],
                [y[8] for y in data]))
            plot_data['action']['x'].append([x[0] for x in s])
            plot_data['action']['y'].append([y[1] for y in s])

            s = sorted(zip(
                [x[10] for x in data],
                [y[11] for y in data]))
            plot_data['loss']['x'].append([x[0] for x in s])
            plot_data['loss']['y'].append([y[1] for y in s])

            s = sorted(zip(
                [x[13] for x in data],
                [y[14] for y in data]))
            plot_data['td_error']['x'].append([x[0] for x in s])
            plot_data['td_error']['y'].append([y[1] for y in s])
    break

# plt.subplot(5, 1, 1)
plt.title('Rewards')
for i in range(len(plot_data['rewards']['x'])):
    plt.plot(plot_data['rewards']['x'][i], plot_data['rewards']['y'][i])
# plt.subplot(5, 1, 2)
# plt.title('eps')
# for i in range(len(plot_data['eps']['x'])):
#     plt.plot(plot_data['eps']['x'][i], plot_data['eps']['y'][i])
# plt.subplot(5, 1, 3)
# plt.title('action')
# for i in range(len(plot_data['action']['x'])):
#     plt.plot(plot_data['action']['x'][i], plot_data['action']['y'][i])
# plt.subplot(5, 1, 4)
# plt.title('loss')
# for i in range(len(plot_data['loss']['x'])):
#     plt.plot(plot_data['loss']['x'][i], plot_data['loss']['y'][i])
# plt.subplot(5, 1, 5)
# plt.title('td_error')
# for i in range(len(plot_data['td_error']['x'])):
#     plt.plot(plot_data['td_error']['x'][i], plot_data['td_error']['y'][i])

# plt.show()
plt.savefig('plot.png', dpi=850)
