import random
import csv
from tqdm import tqdm
import os
import tensorflow as tf


def get_values(path, weight, samples):

    values = []
    time_offset = None
    smoothed = 0

    # begin = datetime.now()
    for e in tf.train.summary_iterator(path):

        if time_offset is None:
            time_offset = e.wall_time

        for v in e.summary.value:
            if v.tag == 'input_info/rewards':
                smoothed = weight * smoothed + (1-weight) * v.simple_value
                values.append((e.wall_time - time_offset, e.step, smoothed))
            #     print(v.simple_value)
            # pass
        # print(e.wall_time)
        # print(e.step)

    # print(f'Loading took {datetime.now() - begin}')

    return sorted(random.sample(values, samples), key=lambda i: i[1])


if __name__ == '__main__':

    for subdir, dirs, files in os.walk('tb_logs'):
        for dir in tqdm(dirs):
            for s, ds, fs in os.walk(f'tb_logs/{dir}'):
                with open(f'reward_plots/{dir}.csv', 'w') as fd:
                    writer = csv.writer(fd)
                    values = get_values(f'tb_logs/{dir}/{fs[0]}', 0.99, 5000)

                    writer.writerow(['time', 'step', 'reward'])

                    for r in values:
                        writer.writerow(list(r))