import logging

import numpy as np

from BaseAttacker import BaseAttacker
from Experience import Experience


class AttackLearner(BaseAttacker):

    def __init__(self, experience: Experience, epsilon=.01, alpha=.05, m=10, downtime=7, train=True):
        super().__init__(m, downtime)
        self.alpha = alpha
        self.epsilon = epsilon
        self.train = train
        self.experience = experience
        self.logger = logging.getLogger(__name__)
        self.time = 0

    def update_utility(self, u):
        super().update_utility(u)
        self.experience.record_reward(u)

    def select_action(self, time):
        self.time = time

        new_state = []
        for server in self.servers:
            up = int(server['status'] == -1)
            time_to_up = 0 if up else server['status'] + self.downtime - time
            progress = server['progress']
            control = server['control']

            if (up and time_to_up != 0) or (not up and (progress != 0 or control == 1)) or (time_to_up > 0 and progress > 0):
                raise Exception('WTF is this state?', server)

            new_state.append(
                [up, time_to_up, progress, control]
            )

        self.experience.record_state(new_state)

        if self.train:
            if np.random.rand() < self.epsilon:
                # action = np.random.randint(0, self.m + 1)
                # action = self.experience.predict_heuristic(new_state)
                if np.random.rand() < .1:
                    action = np.random.randint(0, self.m + 1)
                else:
                    max = -1
                    index = -1

                    for i in range(self.m):
                        if self.servers[i]['control'] == 0 and self.servers[i]['status'] == -1:
                            if self.servers[i]['progress'] > max:
                                index = i
                                max = self.servers[i]['progress']

                    action = index + 1

            else:
                action = self.experience.predict(new_state)
        else:
            action = self.experience.predict(new_state)

        self.experience.record_action(action)

        if self.train:
            self.experience.train_model()

        return action - 1

    def finalize(self, f):
        if f and self.train:
            self.experience.store()
