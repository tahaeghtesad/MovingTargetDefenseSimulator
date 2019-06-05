from BaseAttacker import BaseAttacker
import logging
import random


class UniformAttacker(BaseAttacker):
    def __init__(self, p=1, m=10, downtime=7):
        super().__init__(m, downtime)
        self.logger = logging.getLogger('UniformAttacker')
        self.p = p

    def select_action(self, time):

        if time % self.p != 0:
            return -1

        targets = []
        for i in range(len(self.servers)):
            if self.servers[i]['control'] == 0 and self.servers[i]['status'] == -1:
                targets.append(i)
        return -1 if len(targets) == 0 else targets[random.randint(0, len(targets) - 1)]


class MaxProbeAttacker(BaseAttacker):
    def __init__(self, p=1, m=10, downtime=7):
        super().__init__(m, downtime)
        self.logger = logging.getLogger('MaxProbeAttacker')
        self.p = p

    def select_action(self, time):

        if time % self.p != 0:
            return -1

        max = -1
        index = -1

        for i in range(self.m):
            if self.servers[i]['control'] == 0 and self.servers[i]['status'] == -1:
                if self.servers[i]['progress'] > max:
                    index = i
                    max = self.servers[i]['progress']

        return index


class ControlThresholdAttacker(BaseAttacker):
    def __init__(self, t=.6, p=1, m=10, downtime=7):
        super().__init__(m, downtime)
        self.logger = logging.getLogger('ControlThresholdAttacker')
        self.t = t
        self.p = p
        self.last_action = 0

    def select_action(self, time):

        if time - self.last_action <= self.p:
            return -1

        targets = []
        defender_control_or_down = 0
        for i in range(len(self.servers)):
            if self.servers[i]['control'] == 0:
                if self.servers[i]['status'] == -1:
                    targets.append(i)
                defender_control_or_down += 1

        probe = -1

        if len(self.servers) - defender_control_or_down < self.t * self.m:
            probe = -1 if len(targets) == 0 else targets[random.randint(0, len(targets) - 1)]

        return probe
