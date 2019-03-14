class Experience:
    def __init__(self, dr=.7, max_memory_size=128):
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
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError
