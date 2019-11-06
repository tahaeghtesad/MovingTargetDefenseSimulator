import random
import math
import numpy as np


class BaseDefender:
    def __init__(self, m=10) -> None:
        self.m = m

    def predict(self, obs):
        return -1 + 1

    @staticmethod
    def gen_configurations():
        yield BaseDefender()


class UniformDefender(BaseDefender):
    def __init__(self, p=4, m=10) -> None:
        super().__init__(m)
        self.p = p

        self.counter = 0

    def predict(self, obs):

        self.counter += 1

        if self.counter % self.p:
            return -1 + 1

        targets = []
        for i in range(self.m):
            if obs[i * 5 + 0] == 1:  # and obs[i * 5 + 2] > 0:
                targets.append(i)

        return (-1 if len(targets) == 0 else targets[random.randint(0, len(targets) - 1)]) + 1

    @staticmethod
    def gen_configurations():
        for p in range(1, 14):
            yield UniformDefender(p=p)


class PCPDefender(BaseDefender):

    def __init__(self, p=4, pi=7, m=10) -> None:
        super().__init__(m)
        self.p = p
        self.pi = pi

    def predict(self, obs):
        targets = []
        for i in range(self.m):
            if obs[i * 5 + 2] >= self.pi:  # 0 ~> up, 1 ~> time_to_up, 2 ~> observed_progress, 3 ~> time_since_last_probe, 4 ~> time_since_last_reimage
                targets.append(i)

            if 1 <= obs[i * 5 + 2] and obs[i * 5 + 3] > self.p:
                targets.append(i)

        return (-1 if len(targets) == 0 else targets[random.randint(0, len(targets) - 1)]) + 1

    @staticmethod
    def gen_configurations():
        for p in range(1, 14):
            for pi in range(1, 18, 2):
                yield PCPDefender(p=p, pi=pi)


class ControlThresholdDefender(BaseDefender):
    def __init__(self, tau=0.8, p=4, alpha=.05, m=10):
        super().__init__(m)
        self.tau = tau
        self.alpha = alpha
        self.last_action = 0
        self.p = p
        self.counter = 0

    def predict(self, obs):
        self.counter += 1

        if self.counter - self.last_action <= self.p:
            return -1 + 1

        targets = []
        for i in range(self.m):  # 0 ~> up, 1 ~> time_to_up, 2 ~> observed_progress, 3 ~> time_since_last_probe, 4 ~> time_since_last_reimage
            if obs[i * 5 + 2] != 0:
                prob = (1 - math.exp(-self.alpha * (obs[i * 5 + 2] + 1)))
                targets.append((i, prob))

        reimage = -1
        max = 0
        if sum(1 - t[1] for t in targets) + self.m - len(targets) < self.tau * self.m:
            for i, prob in targets:
                if prob > max:
                    max = prob
                    reimage = i

        if reimage != -1:
            self.last_action = self.counter

        return reimage + 1

    @staticmethod
    def gen_configurations():
        for p in range(1, 14):
            for tau in np.arange(0.8, 1.05, 0.05):
                yield ControlThresholdDefender(p=p, tau=tau)


class MaxProbeDefender(BaseDefender):
    def __init__(self, m=10, p=4):
        super().__init__(m)
        self.p = p
        self.counter = 0

    def predict(self, obs):
        self.counter += 1

        if self.counter % self.p:
            return -1 + 1

        max = 1
        index = -1
        max_count = 0

        for i in range(self.m):  # 0 -> up / 1 -> time to up / 2 -> progress / 3 -> control
            # up, time_to_up, observed_progress, time_since_last_probe, time_since_last_reimage
            if obs[i * 5 + 0] == 1:
                if obs[i * 5 + 2] == max:
                    max_count += 1
                    if random.random() < 1 / max_count:
                        index = i
                if obs[i * 5 + 2] > max:
                    index = i
                    max = obs[i * 5 + 2]
                    max_count = 1

        return index + 1

    @staticmethod
    def gen_configurations():
        for p in range(1, 14):
            yield MaxProbeDefender(p=p)
