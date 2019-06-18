from rl.policy import EpsGreedyQPolicy

import numpy as np


class EpsNoNoOp(EpsGreedyQPolicy):

    def __init__(self, eps=.1):
        super().__init__(eps)
        self.step = 0

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """

        self.step += 1

        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.randint(1, nb_actions)
        else:
            action = np.argmax(q_values)
        return action

    def get_config(self):
        """Return configurations of EpsGreedyPolicy

        # Returns
            Dict of config
        """
        config = super().get_config()
        return config