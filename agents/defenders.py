import random


class BaseDefender:
    def __init__(self, m=10, downtime=7) -> None:
        self.m = m
        self.downtime = downtime

    def predict(self, obs):
        return -1


class UniformDefender(BaseDefender):
    def __init__(self, p=4, m=10, downtime=7) -> None:
        super().__init__(m, downtime)
        self.p = p

        self.counter = 0

    def predict(self, obs):

        self.counter += 1

        if self.counter % self.p:
            return -1

        targets = []
        for i in range(self.m):
            if obs[i * 5 + 0] == 1:
                targets.append(i)

        return -1 if len(targets) == 0 else targets[random.randint(0, len(targets) - 1)]


class PCPDefender(BaseDefender):

    def __init__(self, p=4, pi=7, m=10, downtime=7) -> None:
        super().__init__(m, downtime)
        self.p = p
        self.pi = pi

        self.counter = 0

    def predict(self, obs):
        self.counter += 1

        target = -1
        for i in range(self.m):
            if obs[i * 5 + 2] >= self.pi:
                target = i

            if 1 <= obs[i * 5 + 2] and obs[i * 5 + 3] > self.p:
                target = i

        return target
