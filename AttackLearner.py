import logging
from gym_mtd.envs.MTDEnv import MovingTargetDefenceEnv
from rl.processors import Processor

import numpy as np

from Experience import Experience


class AttackLearner:

    def __init__(self, experience: Experience, env: MovingTargetDefenceEnv, processor: Processor):
        self.alpha = env.alpha
        self.experience = experience
        self.logger = logging.getLogger(__name__)
        self.time = 0

        self.m = env.m
        self.downtime = env.downtime

        self.processor = processor

        self.env = env

    def fit(self, steps, episodes):

        epsilon = 1
        decay = 0.96

        for e in range(episodes):
            observation = self.processor.process_observation(self.env.reset())
            self.experience.record_state(observation)
            for s in range(steps):

                prev_obs = observation

                if np.random.rand() < epsilon * decay ** e:
                    if np.random.rand() < .1:
                        action = np.random.randint(0, self.m + 1)
                    else:
                        max = -1
                        index = -1

                        for i in range(self.m):
                            if observation[i * 4 + 3] == 0 and observation[i * 4 + 0] == 1:
                                if observation[i * 4 + 2] > max:
                                    index = i
                                    max = observation[i * 4 + 2]

                        action = index + 1

                else:
                    action = np.argmax(self.experience.predict(observation))

                observation, reward, done, inf = \
                    self.processor.process_step(
                        * self.env.step(
                            self.processor.process_action(action)))

                if np.array_equal(prev_obs, observation) and action != 0 and observation[(action - 1) * 4 + 3] == 0:
                    print('Exception?')

                self.experience.record_reward(reward)
                self.experience.record_action(action)
                self.experience.record_done(done)
                self.experience.record_state(observation)

                self.experience.train_model(32)

                if done:
                    break
            self.logger.info(f'Episode {e}/{episodes} is done.')  # TODO add statistics.

    def test(self, steps, episodes):
        for e in range(episodes):
            observation = self.processor.process_observation(self.env.reset())
            for s in range(steps):
                action = np.argmax(self.experience.predict(observation))
                observation, reward, done, inf = self.processor.process_step(* self.env.step(self.processor.process_action(action)))

    def finalize(self, f):
        if f:
            self.experience.store()
