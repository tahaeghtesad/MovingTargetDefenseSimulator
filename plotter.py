import csv
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 9))

mode = {
    'target': 'attacker',
    'best_reward': 0.6188,
    'ymin': 0.45,
    'ymax': 0.8
}

# mode = {
#     'target': 'defender',
#     'best_reward': 0.8876,
#     'ymin': 0.75,
#     'ymax': 0.97
# }

def extract(name: str):
    with open('reports.csv') as fd:
        reader = csv.DictReader(fd)

        for data in reader:
            if name.split('_')[2] in data['\ufeffname']:
                # return f"{data['learning_rate']}-{data['gamma']}-{data['layers']}-{data['layer_normalization']}-{data['double_q']}-{data['dueling']}-{data['prioritized_replay']}"
                return data['info']


def get_best_reward(name: str):
    with open('reports.csv') as fd:
        reader = csv.DictReader(fd)

        for data in reader:
            if name.split('_')[2] in data['\ufeffname']:
                return float(data['best_reward'])
    return 0


def smooth(series, weight):
    ret = [sum(series[0:200])/200]
    smoothed = sum(series[0:200])/200
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

for subdir, d, f in os.walk('reward_plots'):
    random_samples = f  #random.sample(f, 20)
    for n in tqdm(random_samples):
        if mode['target'] in n:
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

def add_plot(count, order, data, title):
    # plt.subplot(count, 1, order)
    plt.title('Learning Curve')
    plt.axis([0, 500000, mode['ymin'], mode['ymax']])
    for i in range(len(data['x'])):
        if extract(labels[i]) is not None:
        # if get_best_reward(labels[i]) >= mode['best_reward']:
            if title == 'Epsilon':
                plt.plot(data['x'][i], data['y'][i], label=f'{extract(labels[i])}')
            elif title == 'Rewards':
                plt.plot(data['x'][i], smooth(data['y'][i], 0.0), label=f'{extract(labels[i])}')
            else:
                plt.plot(data['x'][i], smooth(data['y'][i], 0.0), label=f'{extract(labels[i])}')

    plt.legend(loc='lower right', prop={'size': 18})
    plt.grid()


add_plot(1, 1, plot_data['rewards'], 'Rewards')
# add_plot(4, 2, plot_data['action'], 'Action')
# add_plot(4, 3, plot_data['loss'], 'loss')
# add_plot(4, 4, plot_data['td_error'], 'td_error')

plt.savefig(f"plot_{mode['target']}.pgf", bbox_inches='tight')

