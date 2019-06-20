import logging
import os

import numpy as np
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam

from NNExperience import NNExperience


class AttackerNNExperience(NNExperience):
    def __init__(self, name, m=10, dr=.98, max_memory_size=128):
        super().__init__(name, m, dr, max_memory_size)
        self.logger = logging.getLogger(__name__)
        self.model = self.create_model(name, m)

    def create_model(self, name, params):

        # config = tf.ConfigProto(device_count={"CPU": cpu_count()})
        # tensorflow_backend.set_session(tf.Session(config=config))

        model = Sequential()
        model.add(Dense(16, activation='relu', input_shape=(params * 4,)))
        model.add(Dense(16))
        model.add(Dense(16, activation='sigmoid'))
        model.add(Dense(params + 1))
        model.compile(Adam(lr=0.01), 'mse')

        if os.path.isfile(f'{self.name}-weights.h5'):
            logging.info('Loading weight files.')
            model.load_weights(f'{self.name}-weights.h5')

        return model
