import random


class BaseAttacker:
    def __init__(self, m=10, downtime=10):
        self.m = m
        self.downtime = downtime

    def predict(self, obs):
        return -1


class UniformAttacker(BaseAttacker):

    def __init__(self, p=1, m=10, downtime=10):
        super().__init__(m, downtime)
        self.p = p

    def predict(self, obs):

        targets = []
        for i in range(self.m):
            if obs[i * 4 + 3] == 0 and obs[i * 4 + 0] == 1:
                targets.append(i)

        return -1 if len(targets) == 0 else targets[random.randint(0, len(targets) - 1)]


class MaxProbeAttacker(BaseAttacker):

    def __init__(self, p=1, m=10, downtime=10):
        super().__init__(m, downtime)

    def predict(self, obs):
        max = -1
        index = -1

        for i in range(self.m):
            if obs[i * 4 + 3] == 0 and obs[i * 4 + 0] == 1:
                if obs[i * 4 + 2] > max:
                    index = i
                    max = obs[i * 4 + 3]

        return index