from BaseDefender import BaseDefender
import random
import logging
import math
import numpy as np


class UniformDefender(BaseDefender):
    def __init__(self, p=4, m=10, downtime=7):
        super().__init__(m, downtime)
        self.logger = logging.getLogger('UniformDefender')
        self.p = p

    def select_action(self, time, last_probe):
        if time % self.p != 0:
            return -1

        targets = []
        for i in range(len(self.servers)):
            if self.servers[i]['status'] == -1:
                targets.append(i)
        return -1 if len(targets) == 0 else targets[random.randint(0, len(targets) - 1)]


class MaxProbeDefender(BaseDefender):
    def __init__(self, m=10, downtime=7, p=4):
        super().__init__(m, downtime)
        self.logger = logging.getLogger('MaxProbeDefender')
        self.p = p

    def select_action(self, time, last_probe):
        max = 0
        index = -1

        for i in range(self.m):
            if self.servers[i]['status'] == -1:
                if self.servers[i]['progress'] > max:
                    index = i
                    max = self.servers[i]['progress']

        return index if time != 0 and time % self.p == 0 else -1


class PCPDefender(BaseDefender):
    def __init__(self, pi=7, p=4, m=10, downtime=7):
        super().__init__(m, downtime)
        self.logger = logging.getLogger('PCPDefender')
        self.pi = pi
        self.p = p

        self.last_probes = [-1] * self.m

    def select_action(self, time, last_probe):
        if last_probe != -1:
            self.last_probes[last_probe] = time

        target = -1
        for i in range(self.m):
            if self.servers[i]['progress'] >= self.pi:
                target = i

            if 1 <= self.servers[i]['progress'] and self.last_probes[i] + self.p < time:
                target = i

        if target != -1:
            self.last_probes[target] = -1

        return target


class ControlThresholdDefender(BaseDefender):
    def __init__(self, t=.4, p=4, alpha=.05, m=10, downtime=7):
        super().__init__(m, downtime)
        self.logger = logging.getLogger('ControlThresholdDefender')
        self.t = t
        self.alpha = alpha
        self.last_action = 0
        self.p = p

    def select_action(self, time, last_probe):

        if time - self.last_action <= self.p:
            return -1

        targets = []
        for i in range(len(self.servers)):
            if self.servers[i]['progress'] != 0:
                prob = (1 - math.exp(-self.alpha * (self.servers[i]['progress'] + 1)))
                targets.append((i, prob))

        reimage = -1
        max = 0
        if sum(1 - t[1] for t in targets) + self.m - len(targets) < self.t * self.m:
            for i, prob in targets:
                if prob > max:
                    max = prob
                    reimage = i

        if reimage != -1:
            self.last_action = time

        return reimage


class AllKnowingDefender(BaseDefender):
    def __init__(self, m=10, downtime=7):
        super().__init__(m, downtime)
        self.go_up = -1
        self.prev_utility_step = self.m
        self.utility_steps = [1./(1. + math.exp(-5 * (x - 0.5))) for x in np.linspace(0, 1, m + 1)]
        self.last_util = self.utility_steps[self.m]

    def select_action(self, time, last_probe):

        for i in range(len(self.servers)):
            if self.servers[i]['status'] != -1 and time - self.servers[i]['status'] == self.downtime - 1:
                self.go_up = i

        utility_step = self.find_step(self.last_util)

        if utility_step < self.prev_utility_step:
            action = last_probe
        else:
            action = -1

        self.prev_utility_step = utility_step
        if self.go_up != -1:
            self.prev_utility_step += 1

        return action

    def find_step(self, u):
        for i in range(len(self.utility_steps)):
            if self.utility_steps[i] * .99 < u < self.utility_steps[i] * 1.01:
                return i