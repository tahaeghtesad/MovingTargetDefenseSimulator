import numpy as np

import logging


class DefenderProcessor:

    def __init__(self, m=10, downtime=7):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        self.m = m
        self.downtime = downtime

        self.servers = []
        for i in range(m):
            self.servers.append({
                'status': -1,
                'progress': 0
            })

        self.last_probe = [-1] * self.m
        self.last_reimage = [0] * self.m

    ### This method is only called when environment is reset.
    def process_observation(self, observation):
        self.servers = []
        for i in range(self.m):
            self.servers.append({
                'status': -1,
                'progress': 0
            })

        return self.convert_state(0)

    def process_step(self, observation, reward, done, info):

        ### updating reward
        defender_reward = reward['def']
        # assert -2 <= attacker_reward <= 0

        ### extracting observation parameters

        last_probe = observation['def']['last_probe']
        action = observation['att']['action']
        time = observation['time']

        ### updating state

        for s in self.servers:
            if s['status'] != -1:
                if s['status'] + self.downtime <= time:
                    s['status'] = -1

        if not -1 <= last_probe < self.m:
            raise Exception('Out of range server.')

        if last_probe != -1:
            self.servers[last_probe]['progress'] += 1

        if action != -1 and self.servers[action]['status'] == -1:
            self.logger.debug('Reimage was successful.')
            self.servers[action]['status'] = time
            self.servers[action]['progress'] = 0
        else:
            self.logger.debug('Reimage was unsuccessful.')

        if action != 0:
            self.last_reimage[action - 1] = time
            self.last_probe[action - 1] = -1

        ### Converting state
        new_state = self.convert_state(time)

        return new_state, defender_reward - 1, done, info

    def process_action(self, action):
        return action - 1

    def convert_state(self, time):
        new_state = []
        for i, server in enumerate(self.servers):
            up = int(server['status'] == -1)
            time_to_up = 0 if up else server['status'] + self.downtime - time
            observed_progress = server['progress']

            time_since_last_probe = time - self.last_probe[i] if self.last_probe[i] != -1 else -1
            time_since_last_reimage = time - self.last_reimage[i]

            new_state.append(
                [up, time_to_up, observed_progress, time_since_last_probe, time_since_last_reimage]
            )

        return np.array(new_state).flatten()
