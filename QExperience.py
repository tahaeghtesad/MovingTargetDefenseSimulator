from Experience import Experience
import numpy as np


class QExperience(Experience):
    def __init__(self, model, m=10, lr=.3, dr=.7, max_memory_size=128):
        super().__init__(dr, max_memory_size)
        self.model = model
        self.lr = lr
        self.m = m

    def predict(self, state):
        return np.argmax(self.model[repr(state)])

    def record_state(self, state):
        if repr(state) not in self.model:
            self.model[repr(state)] = [0.] * (self.m+1)
        super().record_state(state)

    def train_model(self):
        if len(self.exp) == 0:
            return

        for i in range(self.size):
            last_state, last_action, last_reward, state = self.exp[i]

            q_sa = np.max(self.model[repr(state)])
            q = (1 - self.lr) * self.model[repr(last_state)][last_action] + \
                self.lr * (last_reward + self.dr * q_sa)

            self.model[repr(last_state)][last_action] = q
