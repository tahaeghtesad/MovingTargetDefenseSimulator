from keras.models import Sequential
import numpy as np


class Experience:
    def __init__(self, model: Sequential, dr=.7, max_memory_size=128):
        self.model = model
        self.exp = []
        self.size = max_memory_size

        self.dr = dr

        self.last_state = []
        self.last_reward = 0
        self.last_action = 0

    def record_state(self, state):

        if len(self.last_state) == 0:
            self.last_state = state
        else:
            self.exp.append([self.last_state, self.last_action, self.last_reward, state])
            self.last_state = state
            if len(self.exp) > self.size:
                del self.exp[0]

    def record_action(self, action):
        self.last_action = action

    def record_reward(self, reward):
        self.last_reward = reward

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

