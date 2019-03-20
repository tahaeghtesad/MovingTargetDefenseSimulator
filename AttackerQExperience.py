from QExperience import QExperience
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np


class AttackerQExperience(QExperience):

    def __init__(self, name, delta=7, max_probes=16, m=10, lr=.3, dr=.7, max_memory_size=1000):
        super().__init__(name, m, lr, dr, max_memory_size)
        self.delta = delta
        self.max_probes = max_probes
        self.model = self.create_model(name, (m, delta, max_probes))

    def create_model(self, name, params):
        m, delta, max_probes = params
        model = super().create_model(name, m)
        if model is not None:
            return model
        return np.zeros(((delta + max_probes) * 2,) * m + (m+1,))

    def code(self, state: list) -> tuple:
        ret = ()
        for server in state:
            up, time_to_up, progress, control = server
            progress = progress if progress < self.max_probes else self.max_probes - 1
            calc = control + (-time_to_up + progress + self.delta) * 2
            ret += (calc,)
        return ret

    def decode(self, state: tuple) -> list:
        ret = []
        for i in state:
            control = i % 2
            up = 1 if (i // 2) - self.delta >= 0 else 0
            progress = 0 if not up else (i // 2) - self.delta
            time_to_up = 0 if up else - ((i // 2) - self.delta)
            ret.append([up, time_to_up, progress, control])
        return ret

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = []
        y = []
        z = []
        c = []

        for i in range((self.delta + self.max_probes) * 2):
            for j in range((self.delta + self.max_probes) * 2):
                for k in range(self.m + 1):
                    x.append(i)
                    y.append(j)
                    z.append(self.model[i, j][1])
                    c.append(1)

        ax.scatter(x, y, z, c=c, cmap=plt.hot())
        plt.show()
