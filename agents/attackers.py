import random
import numpy as np


class BaseAttacker:
    def __init__(self, m=10, downtime=7):
        self.m = m
        self.downtime = downtime

    def predict(self, obs):
        return -1 + 1

    @staticmethod
    def gen_configurations():
        yield BaseAttacker()


class UniformAttacker(BaseAttacker):

    def __init__(self, p=1, m=10, downtime=7):
        super().__init__(m, downtime)
        self.p = p

    def predict(self, obs):

        targets = []
        for i in range(self.m):  # 0 -> up / 1 -> time to up / 2 -> progress / 3 -> control
            if obs[i * 4 + 3] == 0 and obs[i * 4 + 0] == 1:
                targets.append(i)

        return (-1 if len(targets) == 0 else targets[random.randint(0, len(targets) - 1)]) + 1

    @staticmethod
    def gen_configurations():
        for p in range(1, 10):
            yield UniformAttacker(p=p)


class MaxProbeAttacker(BaseAttacker):

    def __init__(self, p=1, m=10, downtime=7):
        super().__init__(m, downtime)
        self.p = p
        self.counter = 0

    def predict(self, obs):
        self.counter += 1

        if self.counter % self.p:
            return -1 + 1

        max = 0
        index = -1
        max_count = 0

        for i in range(self.m):  # 0 -> up / 1 -> time to up / 2 -> progress / 3 -> control
            if obs[i * 4 + 3] == 0 and obs[i * 4 + 0] == 1:
                if obs[i * 4 + 2] == max:
                    max_count += 1
                    if random.random() < 1/max_count:
                        index = i
                if obs[i * 4 + 2] > max:
                    index = i
                    max = obs[i * 4 + 2]
                    max_count = 1

        return index + 1

    @staticmethod
    def gen_configurations():
        for p in range(1, 10):
            yield MaxProbeAttacker(p=p)


class ControlThresholdAttacker(BaseAttacker):
    def __init__(self, tau=0.5, p=1, m=10, downtime=7):
        super().__init__(m, downtime)
        self.tau = tau
        self.p = p
        self.last_action = 0
        self.counter = 0

    def predict(self, obs):
        self.counter += 1

        if self.counter - self.last_action <= self.p:
            return -1 + 1

        targets = []
        defender_control_or_down = 0
        for i in range(self.m):  # 0 -> up / 1 -> time to up / 2 -> progress / 3 -> control
            # if self.servers[i]['control'] == 0:
            #     if self.servers[i]['status'] == -1:
            if obs[i * 4 + 3] == 0:
                if obs[i * 4 + 0] == 1:
                    targets.append(i)
                defender_control_or_down += 1

        probe = -1 + 1

        if self.m - defender_control_or_down < self.tau * self.m:
            probe = (-1 if len(targets) == 0 else targets[random.randint(0, len(targets) - 1)]) + 1

        return probe

    @staticmethod
    def gen_configurations():
        for p in range(1, 4):
            for tau in np.arange(0.5, 1., 0.2):
                yield ControlThresholdAttacker(p=p, tau=tau)
