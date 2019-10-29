import csv
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 35))


def extract(name: str):
    with open('reports.csv') as fd:
        reader = csv.DictReader(fd)

        for data in reader:
            if name.split('_')[2] in data['\ufeffname']:
                return f"{data['episodes']}-{data['layers']}-{data['exploration_fraction']}-{data['exploration_final']}"


def smooth(series, weight):
    ret = [series[0]]
    smoothed = series[0]
    for i in range(1, len(series)):
        smoothed = smoothed * weight + (1-weight) * series[i]
        ret.append(smoothed)
    return ret

plot_data = {
    'rewards': {'x': [], 'y': []},
    'eps': {'x': [], 'y': []},
    'action': {'x': [], 'y': []},
    'loss': {'x': [], 'y': []},
    'td_error': {'x': [], 'y': []}
}

labels = []
avg = []

for subdir, d, f in os.walk('reward_plots'):
    random_samples = f  #random.sample(f, 20)
    for n in tqdm(random_samples):
        if 'attacker' in n:
            labels.append(n)
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

                avg.append(sum([y[1] for y in s])/len([y[1] for y in s]))

                s = sorted(zip(
                    [x[4] for x in data],
                    [y[5] for y in data]))
                plot_data['eps']['x'].append([x[0] for x in s])
                plot_data['eps']['y'].append([y[1] for y in s])

                s = sorted(
                            zip(
                                [x[7] for x in data],
                                [y[8] for y in data]
                            )
                        )
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

selected_reward = sorted(avg)[-10]


def add_plot(count, order, data, title):
    plt.subplot(count, 1, order)
    plt.title(title)
    for i in range(len(data['x'])):
        if avg[i] >= selected_reward:
            if title == 'Rewards' or title == 'Epsilon':
                plt.plot(data['x'][i], data['y'][i], label=f'{extract(labels[i])}')
            else:
                plt.plot(data['x'][i], smooth(data['y'][i], 0.99), label=f'{extract(labels[i])}')

    plt.legend(loc='lower right')
    plt.grid()


add_plot(5, 1, plot_data['rewards'], 'Rewards')

add_plot(5, 2, plot_data['eps'], 'Epsilon')
add_plot(5, 3, plot_data['action'], 'Action')
add_plot(5, 4, plot_data['loss'], 'loss')
add_plot(5, 5, plot_data['td_error'], 'td_error')

plt.savefig('plot.png', dpi=300)

