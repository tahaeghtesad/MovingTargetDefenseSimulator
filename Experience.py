class Experience:
    def __init__(self, name, dr=.7, max_memory_size=128):
        self.exp = []
        self.name = name

        self.size = max_memory_size

        self.dr = dr

        self.last_state = []
        self.last_reward = 0
        self.last_action = 0

        self.time = 0

    def record_state(self, state):

        if len(self.last_state) == 0:
            self.last_state = state
        else:
            self.exp.append([self.last_state, self.last_action, self.last_reward, state, self.time])
            self.last_state = state
            if len(self.exp) > self.size:
                del self.exp[0]

        self.time += 1

    def retrieve_exp(self):
        return self.exp

    def store_exp(self, exp):
        self.exp = exp

    def record_action(self, action):
        self.last_action = action

    def record_reward(self, reward):
        self.last_reward = reward

    @staticmethod
    def create_model(name, m):
        raise NotImplementedError

    def store(self):
        raise NotImplementedError

    def predict(self, state):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError
