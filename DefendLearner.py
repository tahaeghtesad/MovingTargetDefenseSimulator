from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.backend import tensorflow_backend

from BaseDefender import BaseDefender
from Experience import Experience

import tensorflow as tf
import numpy as np

import os
import math


class DefendLearner(BaseDefender):

    def __init__(self, model: Sequential = None, epsilon=.01, alpha=.05, m=10, downtime=7):
        super().__init__(m, downtime)
        self.alpha = alpha
        self.epsilon = epsilon
        self.experience = Experience(model if model is not None else DefendLearner.create_model(m))

    def update_utility(self, u):
        super().update_utility(u)
        self.experience.record_reward(u)

    def select_action(self, time, last_probe):

        new_state = []
        for server in self.servers:
            up = int(server['status'] == -1)
            time_to_up = 0 if up else server['status'] + self.downtime - time
            observed_progress = server['progress']
            prob = (1 - math.exp(-self.alpha * (server['progress'] + 1)))

            new_state.append(
                [up, time_to_up, observed_progress, prob]
            )

        self.experience.record_state(new_state)

        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.m + 1)
        else:
            action = self.experience.predict(new_state)

        self.experience.record_action(action)

        if time % 128 == 0:
            self.experience.train_model()

        return action - 1

    @staticmethod
    def create_model(m=10):

        config = tf.ConfigProto(device_count={"CPU": 8})
        tensorflow_backend.set_session(tf.Session(config=config))

        model = Sequential()
        model.add(Dense(m * 4, activation='sigmoid', input_shape=(m, 4, )))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(m + 1, activation='softmax'))
        model.compile('adam', 'mse')

        if os.path.isfile('weights.h5'):
            model.load_weights('weights.h5')

        return model

