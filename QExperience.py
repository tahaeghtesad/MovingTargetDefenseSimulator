from Experience import Experience
import numpy as np

import os
import pickle
import logging
import h5py


class QExperience(Experience):
    def __init__(self, name, m=10, lr=.3, dr=.7, max_memory_size=1000):
        super().__init__(name, dr, max_memory_size)
        self.lr = lr
        self.m = m
        self.model = None

    def predict(self, state):
        return np.argmax(self.model[self.code(state)])

    def record_state(self, state):
        super().record_state(state)

    def create_model(self, name, m):
        if os.path.isfile(f'{name}-weights.h5'):
            logging.info('Loading weight files.')
            with h5py.File(f'{name}-weights.h5', 'r') as dataset:
                return dataset['weights'][:]

    def store(self):
        with h5py.File(f'{self.name}-weights.h5', 'w') as file:
            return file.create_dataset('weights', self.model.shape, 'f', self.model)

    def code(self, state: list) -> tuple:
        raise NotImplementedError

    def decode(self, state: tuple) -> list:
        raise NotImplementedError

    def train_model(self, size=1):
        if len(self.exp) == 0:
            return

        for i in self.exp[-size:]:
            last_state, last_action, last_reward, state, time = i

            q_sa = np.max(self.model[self.code(state)])
            q = (1 - self.lr) * self.model[self.code(last_state)][last_action] + \
                self.lr * (last_reward + self.dr * q_sa)

            self.model[self.code(last_state)][last_action] = q
