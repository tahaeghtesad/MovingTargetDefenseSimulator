import math
import random
from tqdm import tqdm


def sigmoid(x):
    return 1. / (1. + math.exp(-0.05 * x))


min = 2**64
max = 0
sum = 0

dist = [0] * 32

turns = 10000000

for i in tqdm(range(turns)):
    j = 0
    while True:
        if random.random() < sigmoid(j):
            if j > max:
                max = j
            if j < min:
                min = j
            sum += j
            dist[j] += 1
            break
        j += 1

print(f'Min/Max/Avg:{min}/{max}/{sum/turns}')
print(f'Dist: {[d/turns for d in dist]}')
