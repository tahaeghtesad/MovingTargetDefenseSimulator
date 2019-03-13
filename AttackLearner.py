from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.backend import tensorflow_backend

from BaseAttacker import BaseAttacker
from Experience import Experience

import tensorflow as tf
import numpy as np

import os
import math
import logging


class AttackLearner(BaseAttacker):

    def __init__(self, model: Sequential = None, epsilon=.01, alpha=.05, m=10, downtime=7, train=True):
        super().__init__(m, downtime)
        self.alpha = alpha
        self.epsilon = epsilon
        self.model = model if model is not None else AttackLearner.create_model(m)
        self.train=train
        self.experience = Experience(self.model)
        self.logger = logging.getLogger(__name__)

    def update_utility(self, u):
        super().update_utility(u)
        self.experience.record_reward(u)

    def select_action(self, time):

        new_state = []
        for server in self.servers:
            up = int(server['status'] == -1)
            time_to_up = 0 if up else server['status'] + self.downtime - time
            progress = server['progress']
            control = server['control']

            new_state.append(
                [up, time_to_up, progress, control]
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
            self.model.save_weights('attacker-weights.h5')

    @staticmethod
    def create_model(m=10):

        config = tf.ConfigProto(device_count={"CPU": 8})
        tensorflow_backend.set_session(tf.Session(config=config))

        model = Sequential()
        model.add(Dense(m * 4, activation='sigmoid', input_shape=(m, 4, )))
        model.add(Flatten())
        model.add(Dense(m * 64, activation='tanh'))
        model.add(Dense(m + 1))
        model.compile('adam', 'mse')

        if os.path.isfile('attacker-weights.h5'):
            logging.info('Loading weight files.')
            model.load_weights('attacker-weights.h5')

        return model

