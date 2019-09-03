import numpy as np

import logging


class AttackerProcessor:

    def __init__(self, m=10, downtime=7):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        self.m = m
        self.downtime = downtime

        self.servers = []
        for i in range(m):
            self.servers.append({
                'status': -1,
                'progress': 0,
                'control': 0
            })

    ### This method is only called when environment is reset.
    def process_observation(self, observation):
        self.servers = []
        for i in range(self.m):
            self.servers.append({
                'status': -1,
                'progress': 0,
                'control': 0
            })

        return self.convert_state(0)

    def process_step(self, observation, reward, done, info):

        ### updating reward
        attacker_reward = reward['att']# - reward['def']
        # assert -2 <= attacker_reward <= 0

        ### extracting observation parameters

        success = observation['att']['success']
        last_reimage = observation['att']['last_reimage']
        action = observation['att']['action']
        time = observation['time']

        ### updating state

        for s in self.servers:
            if s['status'] != -1:
                if s['status'] + self.downtime <= time:
                    s['status'] = -1

        if not -1 <= last_reimage < self.m:
            raise Exception('Out of range server.')

        if last_reimage != -1:
            self.servers[last_reimage] = {
                'status': time,
                'control': 0,
                'progress': 0
            }

        if action != -1:
            if success == 1:
                self.servers[action]['control'] = 1
                # self.servers[action]['progress'] += 1
                self.servers[action]['status'] = -1
                self.logger.debug('Probe was successful.')
            elif success == 0:
                self.servers[action]['progress'] += 1
                self.servers[action]['control'] = 0
                self.servers[action]['status'] = -1
                self.logger.debug('Probe was unsuccessful.')
            elif success == -1:
                self.servers[action]['status'] = self.servers[action]['status'] if self.servers[action][
                                                                                       'status'] != -1 else time
                self.servers[action]['progress'] = 0
                self.servers[action]['control'] = 0
                self.logger.debug('Server was down.')


        ### Converting state
        new_state = self.convert_state(time)

        return new_state, attacker_reward, done, info

    def process_action(self, action):
        return action - 1

    def convert_state(self, time):
        new_state = []
        for server in self.servers:
            up = int(server['status'] == -1)
            time_to_up = 0 if up else server['status'] + self.downtime - time
            progress = server['progress']
            control = server['control']

            new_state.append(
                [up, time_to_up, progress, control]
            )
        return np.array(new_state).flatten()
