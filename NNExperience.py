from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv3D, BatchNormalization
from keras.backend import tensorflow_backend

from Experience import Experience

from keras.models import Sequential
import numpy as np

import random
import logging
import os


class NNExperience(Experience):
    def __init__(self, m=10, dr=.7, max_memory_size=128, episode_length=1000):
        super().__init__(dr, max_memory_size)
        self.model: Sequential = None
        self.episode_length = episode_length
        self.logger = logging.getLogger(__name__)
        self.model = self.create_model(m)

    def predict(self, state):
        np_state = np.array(state)
        return np.argmax(self.model.predict(np.array([state]))[0])

    def create_model(self, name, params):

        # config = tf.ConfigProto(device_count={"CPU": 8})
        # tensorflow_backend.set_session(tf.Session(config=config))

        model = Sequential()
        model.add(Dense(64, activation='sigmoid', input_shape=(m, 4)))
        model.add(Flatten())
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(m + 1))
        model.compile('adam', 'mse')

        if os.path.isfile(f'{name}-weights.h5'):
            logging.info('Loading weight files.')
            model.load_weights('attacker-weights.h5')

        return model

    def store(self, name):
        self.model.save_weights(f'{name}-weights.h5')

    def train_model(self, size):
        if len(self.exp) < size:
            return

        samples = random.sample(self.exp, size)

        states = [c[0] for c in samples]
        next_states = [c[3] for c in samples]
        trainings = self.model.predict(np.array(states))
        q_sas = self.model.predict(np.array(next_states))

        for i in range(size):
            if samples[i][4] == self.episode_length - 1:
                trainings[i][samples[i][1]] = samples[i][2]
            else:
                q_sa = np.max(q_sas[i])
                trainings[i][samples[i][1]] = samples[i][2] + self.dr * q_sa

        h = self.model.fit(np.array(states), trainings, epochs=1, batch_size=size, verbose=0)
        self.logger.debug(f'Loss: {h.history["loss"][0]}')
