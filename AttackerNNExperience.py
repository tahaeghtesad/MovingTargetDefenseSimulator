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

    def create_model(self, name, params):

        # config = tf.ConfigProto(device_count={"CPU": cpu_count()})
        # tensorflow_backend.set_session(tf.Session(config=config))

        model = Sequential()
        model.add(Dense(64, activation='sigmoid', input_shape=(params, 4)))
        model.add(Flatten())
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(params + 1))
        model.compile('adam', 'mse')

        if os.path.isfile(f'{self.name}-weights.h5'):
            logging.info('Loading weight files.')
            model.load_weights(f'{self.name}-weights.h5')

        return model
