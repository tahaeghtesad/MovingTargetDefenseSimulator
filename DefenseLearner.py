from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.backend import tensorflow_backend

from BaseDefender import BaseDefender
from Experience import Experience

import tensorflow as tf
import numpy as np

import random
import os
import math
import logging


class DefenseLearner(BaseDefender):

    def __init__(self, experience: Experience = None, epsilon=.01, alpha=.05, m=10, downtime=7, train=True):
        super().__init__(m, downtime)
        self.alpha = alpha
        self.epsilon = epsilon
        self.train = train
        self.experience = experience
        self.logger = logging.getLogger(__name__)

        self.last_probe = [-1] * self.m
        self.last_reimage = [0] * self.m

    def update_utility(self, u):
        super().update_utility(u)
        self.experience.record_reward(u)

    def select_action(self, time, last_probe):

        if last_probe != -1:
            self.last_probe[last_probe] = time

        new_state = []
        for i, server in enumerate(self.servers):
            up = int(server['status'] == -1)
            time_to_up = 0 if up else server['status'] + self.downtime - time
            observed_progress = server['progress']
            prob = (1 - math.exp(-self.alpha * (server['progress'] + 1)))  # probability of attacker controlling that server

            time_since_last_probe = time - self.last_probe[i] if self.last_probe[i] != -1 else -10  # Why -10?
            time_since_last_reimage = time - self.last_reimage[i]

            new_state.append(
                [up, time_to_up, observed_progress, time_since_last_probe]
            )

        self.experience.record_state(new_state)

        if self.train:
            if np.random.rand() < self.epsilon:
                # action = np.random.randint(0, self.m + 1)
                action = -1
                for i in range(self.m):
                    if self.servers[i]['progress'] >= 4:
                        action = i

                    if 1 <= self.servers[i]['progress'] and self.last_probe[i] + 7 < time:
                        action = i

                action += 1
            else:
                action = self.experience.predict(new_state)
        else:
            action = self.experience.predict(new_state)

        self.experience.record_action(action)

        if self.train:
            self.experience.train_model()

        if action != 0:
            self.last_reimage[action - 1] = time
            self.last_probe[action - 1] = -1

        return action - 1

    def finalize(self, f):
        if f and self.train:
            self.experience.store()
