from keras.models import Sequential
import numpy as np


class Experience:
    def __init__(self, model: Sequential, dr=.9, max_memory_size=128):
        self.model = model
        self.exp = []
        self.size = max_memory_size

        self.dr = dr

        self.last_state = []
        self.last_reward = 0
        self.last_action = 0

    def record_state(self, state):

        if len(self.exp) == 0:
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

    def train_model(self, size=128):
        if len(self.exp) == 0:
            return

        samples = np.random.choice(self.exp, size)

        states = []
        trainings = []
        for sample in samples:
            q_sa = np.max(self.model.predict(sample[3])[0])
            q = self.model.predict(sample[0])[0]
            q[samples[1]] = sample[2] + self.dr * q_sa

            states.append(sample[3])
            trainings.append(q)

        self.model.fit(np.array(states), np.array(trainings), verbose=0)
        self.model.save_weights('weights.h5')
