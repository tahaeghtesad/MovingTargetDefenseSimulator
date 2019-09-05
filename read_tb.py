import random
import csv
import multiprocessing
import os
import tensorflow as tf


def get_values(path, weight, sample_no):

    values = []
    time_offset = None
    smoothed = {
        'reward': 0,
        'eps': 0,
        'action': 0,
        'loss': 0,
        'td_error': 0
    }

    # begin = datetime.now()
    for e in tf.train.summary_iterator(path):

        if time_offset is None:
            time_offset = e.wall_time

        for v in e.summary.value:
            if v.tag == 'input_info/rewards':
                smoothed['reward'] = weight * smoothed['reward'] + (1-weight) * v.simple_value
            if v.tag == 'input_info/eps':
                smoothed['eps'] = v.simple_value
            if v.tag == 'game/actions':
                smoothed['action'] = weight * smoothed['action'] + (1-weight) * v.simple_value
            if v.tag == 'loss/loss':
                smoothed['loss'] = weight * smoothed['loss'] + (1-weight) * v.simple_value
            if v.tag == 'loss/td_error':
                smoothed['td_error'] = weight * smoothed['td_error'] + (1 - weight) * v.simple_value

            values.append([e.wall_time - time_offset, e.step] + smoothed.values())

        # print(e.wall_time)
        # print(e.step)

    # print(f'Loading took {datetime.now() - begin}')

    samples = random.sample(values, sample_no)

    return samples


def store(params):
    dir, name = params
    values = get_values(f'tb_logs/{dir}/{name}', 0.99, 5000)

    with open(f'reward_plots/{dir}.csv', 'w') as fd:
        writer = csv.writer(fd)
        writer.writerow(['time', 'step', 'reward', 'eps', 'action', 'loss', 'td_error'])

        for r in values:
            writer.writerow(list(r))

    print(f'{dir} Compiled.')
    return None


if __name__ == '__main__':

    paths = []

    for subdir, dirs, files in os.walk('tb_logs'):
        for dir in dirs:
            for s, ds, fs in os.walk(f'tb_logs/{dir}'):
                paths.append((dir, fs[0]))

    pool = multiprocessing.Pool(int(multiprocessing.cpu_count() / 4))
    pool.map(store, paths)
