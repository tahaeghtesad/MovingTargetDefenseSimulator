from stable_baselines import DQN


class DQNWrapper(DQN):
    def predict(self, observation, state=None, mask=None, deterministic=True):
        return super().predict(observation, state, mask, deterministic)[0]
