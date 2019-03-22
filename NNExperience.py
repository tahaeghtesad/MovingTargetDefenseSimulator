from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv3D, BatchNormalization
from keras.backend import tensorflow_backend

from Experience import Experience
import numpy as np

import random
import logging


class NNExperience(Experience):
    def __init__(self, name, m, dr, max_memory_size):
        super().__init__(name, dr, max_memory_size)
        self.model: Sequential = None
        self.logger = logging.getLogger(__name__)
        self.m = m

    def predict(self, state):
        return np.argmax(self.model.predict(np.array([state]))[0])

    def create_model(self, name, params):
        raise NotImplementedError

    def store(self):
        self.model.save_weights(f'{self.name}-weights.h5')

    def train_model(self, size=16):
        if len(self.exp) < size:
            return

        if self.exp[-1][-1] % self.size != 0:
            return

        # samples = random.sample(self.exp, size)
        samples = self.exp

        states = [c[0] for c in samples]
        next_states = [c[3] for c in samples]
        trainings = self.model.predict(np.array(states))
        q_sas = self.model.predict(np.array(next_states))

        for i in range(len(samples)):
            q_sa = np.max(q_sas[i])
            trainings[i][samples[i][1]] = samples[i][2] + self.dr * q_sa

        h = self.model.fit(np.array(states), trainings, epochs=1, batch_size=size, verbose=0)
        self.logger.debug(f'Loss: {h.history["loss"][0]}')
