import math
import random
import time
from tqdm import tqdm


def sigmoid(x):
    return (1 - math.exp(
        -0.05 * (x + 1)))


min = 2**64
max = 0
ssum = 0

dist = [0] * 32

turns = 100000

for i in tqdm(range(turns)):
    j = 0
    while True:
        if random.random() < sigmoid(j):
            if j > max:
                max = j
            if j < min:
                min = j
            ssum += j
            dist[j] += 1
            break
        j += 1
time.sleep(.1)
print(f'Min/Max/Avg:{min}/{max}/{ssum / turns}')
print(f'Dist: {[d/turns for d in dist]}')

cdf = [sum(dist[:i+1]) for i in range(32)]

print(f'CDF: {[i/turns for i in cdf]}')
