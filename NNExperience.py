from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv3D, BatchNormalization
from keras.backend import tensorflow_backend
from keras.callbacks import Callback
import matplotlib.pyplot as plt
from scipy.special import softmax

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
        self.lossHistory = LossHistory()

    def predict(self, state):
        values = self.model.predict(np.array([state]))[0]
        return softmax([c * 5 for c in values])

    def create_model(self, name, params):
        raise NotImplementedError

    def store(self):
        self.model.save_weights(f'{self.name}-weights.h5')

    def train_model(self, size=16):
        if len(self.exp) == 0:
            return
        elif len(self.exp) < size:
            samples = self.exp
        else:
            samples = random.sample(self.exp, size)

        states = [c[0] for c in samples]
        next_states = [c[3] for c in samples]
        rewards = [c[2] for c in samples]
        actions = [c[1] for c in samples]

        trainings = self.model.predict(np.array(states))
        q_sas = self.model.predict(np.array(next_states))

        for i in range(len(samples)):
            q_sa = np.max(q_sas[i])
            trainings[i][actions[i]] = rewards[i] + self.dr * q_sa #if rewards[i] > -0.8 else rewards[i]

        h = self.model.fit(np.array(states), trainings, epochs=1, batch_size=size, verbose=0, callbacks=[self.lossHistory])
        self.logger.debug(f'Loss: {h.history["loss"][0]}')


class LossHistory(Callback):

    def __init__(self):
        super().__init__()
        self.losses = []

    def on_train_begin(self, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses.append(logs.get('loss'))

    def report(self):
        nparr = np.array(self.losses)
        # nparr.sort()
        rep = {
            'median': np.median(nparr),
            'max': np.max(nparr),
            'min': np.min(nparr),
            'avg': np.mean(nparr),
            'var': np.var(nparr),
            'std': np.std(nparr)
        }
        self.losses = []
        # plt.plot(nparr)
        # plt.show()
        return rep