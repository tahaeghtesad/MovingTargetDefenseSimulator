from Experience import Experience

from keras.models import Sequential
import numpy as np


class NNExperience(Experience):
    def __init__(self, model: Sequential, dr=.7, max_memory_size=128):
        super().__init__(dr, max_memory_size)
        self.model = model

    def predict(self, state):
        return np.argmax(self.model.predict(np.array([state]))[0])

    def train_model(self):
        if len(self.exp) == 0:
            return

        states = [c[0] for c in self.exp]
        next_states = [c[3] for c in self.exp]
        trainings = self.model.predict(np.array(states))
        q_sas = self.model.predict(np.array(next_states))

        for i in range(self.size):
            q_sa = np.max(q_sas[i])
            trainings[i][self.exp[i][1]] = self.exp[i][2] + self.dr * q_sa

        self.model.fit(np.array(states), trainings, epochs=8, batch_size=self.size, verbose=0)
