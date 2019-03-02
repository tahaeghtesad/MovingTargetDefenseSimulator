from BaseDefender import BaseDefender
import random
import logging
import math


class UniformDefender(BaseDefender):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('UniformDefender')

    def select_action(self, time, last_probe):
        targets = []
        for i in range(len(self.servers)):
            if self.servers[i]['status'] == -1:
                targets.append(i)
        return targets[random.randint(0, len(targets) - 1)]


class MaxProbeDefender(BaseDefender):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('MaxProbeDefender')

    def select_action(self, time, last_probe):
        max = -1
        index = -1

        for i in range(self.m):
            if self.servers[i]['status'] == -1:
                if self.servers[i]['progress'] > max:
                    index = i
                    max = self.servers[i]['progress']

        return index


class PCPDefender(BaseDefender):
    def __init__(self, pi=1, p=1):
        super().__init__()
        self.logger = logging.getLogger('PCPDefender')
        self.pi = pi
        self.p = p

        self.last_probes = [-1] * self.m

    def select_action(self, time, last_probe):
        if last_probe != -1:
            self.last_probes[last_probe] = time

        target = -1
        for i in range(self.m):
            if self.servers[i]['progress'] > self.pi:
                target = i

            if 1 <= self.servers[i]['progress'] and self.last_probes[i] + self.p > time:
                target = i

        if target != -1:
            self.last_probes[target] = -1

        return target


class ControlThresholdDefender(BaseDefender):
    def __init__(self, t=.1, alpha=.05):
        super().__init__()
        self.logger = logging.getLogger('ControlThresholdDefender')
        self.t = t
        self.alpha = alpha
        self.last_action = -1

    def select_action(self, time, last_probe):

        if self.last_action != -1:
            self.last_action = -1
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

        self.last_action = reimage
        return reimage


class ControlTargetDefender(BaseDefender):
    def __init__(self, t=.1, alpha=.05):
        super().__init__()
        self.logger = logging.getLogger('ControlTargetDefender')
        self.t = t
        self.alpha = alpha

    def select_action(self, time, last_probe):
        raise NotImplementedError()
