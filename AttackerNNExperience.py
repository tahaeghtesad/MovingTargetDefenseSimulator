from NNExperience import NNExperience

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv3D, BatchNormalization
from keras.backend import tensorflow_backend
from multiprocessing import cpu_count

import tensorflow as tf
import numpy as np
import os
import logging


class AttackerNNExperience(NNExperience):
    def __init__(self, name, m=10, dr=.9, max_memory_size=128):
        super().__init__(name, m, dr, max_memory_size)
        self.logger = logging.getLogger(__name__)
        self.model = self.create_model(name, m)

        self.heuristic_model = AttackerNNExperience.create_without_cost_heuristic(m)

    @staticmethod
    def create_without_cost_heuristic(m):
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(m, 4)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(m + 1, activation='softmax'))
        model.compile('adam', 'mse')

        if os.path.isfile('attacker-weights-256-128-working.h5'):
            logging.info('Loading ehuristic weight files.')
            model.load_weights('attacker-weights-256-128-working.h5')

        return model

    def predict_heuristic(self, state):
        weights = self.heuristic_model.predict(np.array([state]))[0]
        action = np.random.choice(np.arange(0, self.m + 1), p=weights)
        return action

    def create_model(self, name, params):

        # config = tf.ConfigProto(device_count={"CPU": cpu_count()})
        # tensorflow_backend.set_session(tf.Session(config=config))

        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(params, 4)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(params + 1))
        model.compile('adam', 'mse')

        if os.path.isfile(f'{self.name}-weights.h5'):
            logging.info('Loading weight files.')
            model.load_weights(f'{self.name}-weights.h5')

        return model
