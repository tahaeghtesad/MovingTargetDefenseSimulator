from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.backend import tensorflow_backend

from BaseDefender import BaseDefender
from NNExperience import NNExperience

import tensorflow as tf
import numpy as np

import os
import math
import logging


class DefenseLearner(BaseDefender):

    def __init__(self, model: Sequential = None, epsilon=.01, alpha=.05, m=10, downtime=7, train=True):
        super().__init__(m, downtime)
        self.alpha = alpha
        self.epsilon = epsilon
        self.model = model if model is not None else DefenseLearner.create_neural_model(m)
        self.train = train
        self.experience = NNExperience(self.model)
        self.logger = logging.getLogger(__name__)

    def update_utility(self, u):
        super().update_utility(u)
        self.experience.record_reward(u)

    def select_action(self, time, last_probe):

        new_state = []
        for server in self.servers:
            up = int(server['status'] == -1)
            time_to_up = 0 if up else server['status'] + self.downtime - time
            observed_progress = server['progress']
            prob = (1 - math.exp(-self.alpha * (server['progress'] + 1)))  # probability of attacker controlling that server

            new_state.append(
                [up, time_to_up, observed_progress]
            )

        self.experience.record_state(new_state)

        if self.train:
            if np.random.rand() < self.epsilon:
                action = np.random.randint(0, self.m + 1)
            else:
                action = self.experience.predict(new_state)
        else:
            action = self.experience.predict(new_state)

        self.experience.record_action(action)

        if self.train and time % 128 == 0:
            self.logger.debug('Training...')
            self.experience.train_model()

        return action - 1

    def finalize(self, f):
        if f and self.train:
            self.model.save_weights('defender-weights.h5')
