import gym
from gym.spaces import *
import logging
from enum import Enum
import math
import random
import numpy as np
import time
from Defenders import UniformDefender


class Party(Enum):
    Attacker = 1
    Defender = 2


class MovingTargetDefenceEnv(gym.Env):

    def __init__(self, m=10, downtime=7, alpha=.05, time_limit=1000, probe_detection=0., utenv=0, setting=0, ca=.2, defender=UniformDefender):
        self.logger = logging.getLogger(__name__)
        self.servers = []
        for i in range(m):
            self.servers.append({
                'control': Party.Defender,
                'status': -1,  # -1 means that it is up, a positive number is the time of re-image
                'progress': 0
            })

        self.m = m
        self.downtime = downtime
        self.alpha = alpha
        self.time_limit = time_limit
        self.probe_detection = probe_detection
        self.ca = ca

        self.last_probe = -1  # -1 means that the last probe is not detectable
        self.last_reimage = -1  # -1 means that the last reimage is not detectable
        self.last_attack_cost = 0

        self.utenv, self.setting = MovingTargetDefenceEnv.get_params(utenv, setting)

        self.time = 0
        self.epoch = 0

        self.defender = defender(4, m, downtime)

        self.attacker_servers = []
        for i in range(m):
            self.attacker_servers.append({
                'status': -1,
                'progress': 0,
                'control': 0
            })

        self.action_space = Discrete(m + 1)
        self.observation_space = MultiDiscrete([2, 7, 32, 2] * m) ## for attacker


    @staticmethod
    def sigmoid(x, tth, tsl=5):
        return 1. / (1. + math.exp(-tsl * (x - tth)))

    @staticmethod
    def get_params(env,
                   setting):  # env = [1: control/avail 2: control/config 3:disrupt/avail 4:disrupt/confid] - setting = [0:low 1:major 2:high] #5: my setting!
        utenv = [(1, 1), (1, 0), (0, 1), (0, 0)]
        setenv = [(.2, .2, .2, .2), (.5, .5, .5, .5), (.8, .8, .8, .8), (.2, .2, .8, .8)]
        return utenv[env], setenv[setting]

    def utility(self, nc, nd, w, tth_1, tth_2):
        return w * MovingTargetDefenceEnv.sigmoid(nc / self.m, tth_1) + (1 - w) * MovingTargetDefenceEnv.sigmoid((nc + nd) / self.m, tth_2)

    def probe(self, server):
        if not -1 <= server < self.m:
            raise Exception('Chosen server is not in range', server)

        if server == -1:
            self.last_attack_cost = 0
            return 0

        self.last_attack_cost = -self.ca

        if self.servers[server]['status'] == -1:
            self.servers[server]['progress'] += 1
            if random.random() >= self.probe_detection:
                self.last_probe = server
            else:
                self.last_probe = -1

            if self.servers[server]['control'] == Party.Attacker:
                return 1  # Attacker already had that server

            if random.random() < (1 - math.exp(
                    -self.alpha * (self.servers[server]['progress'] + 1))):  # 1 - e^-alpha*(rho + 1)
                self.servers[server]['control'] = Party.Attacker
                return 1  # Attacker now controls the server
            else:
                return 0  # Attacker just probed a server

        return -1  # The server was down

    def reimage(self, server):
        if not -1 <= server < self.m:
            raise Exception('Chosen server is not in range', server)

        if server == -1:
            self.last_reimage = -1
            return

        if self.servers[server]['status'] != -1:
            return

        if self.servers[server]['control'] == Party.Attacker:
            self.last_reimage = server
        else:
            self.last_reimage = -1

        self.servers[server] = {
            'control': Party.Defender,
            'status': self.time,  # -1 means that it is up, a positive number is the time of re-image
            'progress': 0
        }

    def update_attacker_state(self, action, success):
        ### updating state

        self.logger.debug(f'Selecting server {action} to probe.')

        for s in self.attacker_servers:
            if s['status'] != -1:
                if s['status'] + self.downtime <= self.time:
                    s['status'] = -1

        if not -1 <= self.last_reimage < self.m:
            raise Exception('Out of range server.')

        if self.last_reimage != -1:
            self.attacker_servers[self.last_reimage] = {
                'status': self.time,
                'control': 0,
                'progress': 0
            }

        if action == -1:
            self.logger.debug('Choosing no server to probe.')
            return

        if success == 1:
            self.attacker_servers[action]['control'] = 1
            self.attacker_servers[action]['progress'] += 1
            self.attacker_servers[action]['status'] = -1
            self.logger.debug('Probe was successful.')
        elif success == 0:
            self.attacker_servers[action]['progress'] += 1
            self.attacker_servers[action]['control'] = 0
            self.attacker_servers[action]['status'] = -1
            self.logger.debug('Probe was unsuccessful.')
        elif success == -1:
            self.attacker_servers[action]['status'] = self.attacker_servers[action]['status'] if self.attacker_servers[action]['status'] != -1 else self.time
            self.attacker_servers[action]['progress'] = 0
            self.attacker_servers[action]['control'] = 0
            self.logger.debug('Server was down.')

    def get_attacker_state(self):
        new_state = []
        for server in self.attacker_servers:
            up = int(server['status'] == -1)
            time_to_up = 0 if up else server['status'] + self.downtime - self.time
            progress = server['progress']
            control = server['control']

            if (up and time_to_up != 0) or (not up and (progress != 0 or control == 1)) or (
                    time_to_up > 0 and progress > 0):
                raise Exception('WTF is this state?', server)

            new_state.append(
                [up, time_to_up, progress, control]
            )

        return np.array(new_state).flatten()

    def step(self, action):
        action -= 1
        assert self.time <= self.time_limit
        self.logger.debug(f'Round {self.time}/{self.time_limit}')
        ### Onlining servers

        for i in range(self.m):
            if self.servers[i]['status'] != -1:
                if self.servers[i]['status'] + self.downtime <= self.time:
                    self.servers[i]['status'] = -1

        ### Doing actions

        p_last_probe = self.last_probe

        success = self.probe(action)
        self.update_attacker_state(action, success)

        self.defender.reimage(self.time, p_last_probe, self.reimage)

        ### Calculate utility
        nca = sum(server['control'] == Party.Attacker for server in self.servers)
        ncd = sum(server['control'] == Party.Defender and server['status'] == -1 for server in self.servers)
        nd = sum(server['status'] > -1 for server in self.servers)

        au = self.utility(nca, nd, self.utenv[0], self.setting[0], self.setting[1]) + self.last_attack_cost
        du = self.utility(ncd, nd, self.utenv[1], self.setting[2], self.setting[3])

        self.defender.update_utility(du)

        self.time += 1

        self.logger.debug(f'Received {au} utility.')
        done = self.time == self.time_limit

        #observation, reward, done, info
        return self.get_attacker_state().flatten(), au, done, {}

    def reset(self):
        self.servers = []
        for i in range(self.m):
            self.servers.append({
                'control': Party.Defender,
                'status': -1,  # -1 means that it is up, a positive number is the time of re-image
                'progress': 0
            })

        self.last_probe = -1  # -1 means that the last probe is not detectable
        self.last_reimage = -1  # -1 means that the last reimage is not detectable
        self.last_attack_cost = 0

        self.time = 0
        self.epoch = 0

        self.defender = UniformDefender(4, self.m, self.downtime)

        self.attacker_servers = []
        for i in range(self.m):
            self.attacker_servers.append({
                'status': -1,
                'progress': 0,
                'control': 0
            })

        return self.get_attacker_state()

    def render(self, mode='human'):
        self.logger.warning(f'LastProbe/LastReimage: {self.last_probe}/{self.last_reimage}')
        self.logger.warning(f'Server States: {self.servers}')
        self.logger.warning(f'Attacker View: {self.get_attacker_state()}')
        time.sleep(5)
