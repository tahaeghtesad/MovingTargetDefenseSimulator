import random
import csv
import sys
import os
import time
from datetime import datetime
import tensorflow as tf


def get_values(path, weight, sample_no):

    start = datetime.now()
    print(f'Beginning collection for {path}')

    time_offset = None
    smoothed = {
        'reward': 0,
        'eps': 0,
        'action': 0,
        'loss': 0,
        'td_error': 0
    }

    values = {
        'reward': [],
        'eps': [],
        'action': [],
        'loss': [],
        'td_error': []
    }

    # begin = datetime.now()
    for e in tf.train.summary_iterator(path):

        if time_offset is None:
            time_offset = e.wall_time

        for v in e.summary.value:
            if v.tag == 'input_info/rewards':
                smoothed['reward'] = weight * smoothed['reward'] + (1-weight) * v.simple_value
                values['reward'].append([e.wall_time - time_offset, e.step, smoothed['reward']])
            if v.tag == 'input_info/eps':
                smoothed['eps'] = v.simple_value
                values['eps'].append([e.wall_time - time_offset, e.step, smoothed['eps']])
            if v.tag == 'game/actions':
                smoothed['action'] = weight * smoothed['action'] + (1-weight) * v.simple_value
                values['action'].append([e.wall_time - time_offset, e.step, smoothed['action']])
            if v.tag == 'loss/loss':
                smoothed['loss'] = weight * smoothed['loss'] + (1-weight) * v.simple_value
                values['loss'].append([e.wall_time - time_offset, e.step, smoothed['loss']])
            if v.tag == 'loss/td_error':
                smoothed['td_error'] = weight * smoothed['td_error'] + (1 - weight) * v.simple_value
                values['td_error'].append([e.wall_time - time_offset, e.step, smoothed['td_error']])

    samples = {}
    for k in values.keys():
        samples[k] = random.sample(values[k], sample_no)

    print(f'Done collection for {path}. It took: {datetime.now() - start}')

    return samples


def store(params):
    dir, name = params
    sample_no = 10000
    try:
        values = get_values(f'tb_logs/{dir}/{name}', 0.99, sample_no)

        with open(f'reward_plots/{dir}.csv', 'w') as fd:
            writer = csv.writer(fd)
            header = []
            for k in values.keys():
                header += [f'time_{k}', f'step_{k}', k]

            writer.writerow(header)

            for i in range(0, len(values[next(iter(values))])):
                row = []
                for k in values.keys():
                    row += values[k][i]
                writer.writerow(row)
    except KeyboardInterrupt:
        exit(-1)
    except:
        pass

    return None


dir = sys.argv[1]
name = sys.argv[2]

store((dir, name))

time.sleep(60)