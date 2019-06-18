import gym
from gym.spaces import *
import logging
from enum import Enum
import math
import random
import numpy as np
import time
import copy
from Defenders import UniformDefender


class Party(Enum):
    Attacker = 1
    Defender = 2


class MovingTargetDefenceEnv(gym.Env):

    def __init__(self, m=10, downtime=7, alpha=.05, time_limit=1000, probe_detection=0., utenv=0, setting=0, ca=.2,
                 defender=None):
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

        self.defender = UniformDefender(4, m, downtime) if defender is None else defender
        self.defender_t = copy.deepcopy(defender)

        self.attacker_servers = []
        for i in range(m):
            self.attacker_servers.append({
                'status': -1,
                'progress': 0,
                'control': 0
            })

        self.action_space = Discrete(m + 1)
        self.observation_space = MultiDiscrete([2, 7, 32, 2] * m)  ## for attacker

        self.attacker_total_reward = 0
        self.defender_total_reward = 0

        self.defender_last_action = -1

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
        return w * MovingTargetDefenceEnv.sigmoid(nc / self.m, tth_1) + (1 - w) * MovingTargetDefenceEnv.sigmoid(
            (nc + nd) / self.m, tth_2)

    def probe(self, server):
        if not -1 <= server < self.m:
            raise Exception('Chosen server is not in range', server)

        if server == -1:
            self.last_attack_cost = 0
            self.last_probe = -1
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
            raise Exception('Chosen server is not in range')

        self.defender_last_action = server

        if server == -1:
            return

        if self.servers[server]['status'] != -1:
            return

        if self.servers[server]['control'] == Party.Attacker:
            self.last_reimage = server
        else:
            self.last_reimage = -1

        # Defender reimaged a server which attacker probed in last action: Don't tell defender that attacker even attacked!
        if self.last_probe == server:
            self.last_probe = -1

        self.servers[server] = {
            'control': Party.Defender,
            'status': self.time,  # -1 means that it is up, a positive number is the time of re-image
            'progress': 0
        }

    def step(self, action):
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

        self.defender.reimage(self.time, p_last_probe, self.reimage)

        ### Calculate utility
        nca = sum(server['control'] == Party.Attacker for server in self.servers)
        ncd = sum(server['control'] == Party.Defender and server['status'] == -1 for server in self.servers)
        nd = sum(server['status'] > -1 for server in self.servers)

        assert nca + ncd + nd == self.m, "N_ca, N_cd, or N_d is calculated incorrectly!"

        au = self.utility(nca, nd, self.utenv[0], self.setting[0], self.setting[1]) + self.last_attack_cost
        du = self.utility(ncd, nd, self.utenv[1], self.setting[2], self.setting[3])

        self.defender.update_utility(du)

        self.time += 1

        self.logger.debug(f'Received {au} utility.')
        # done = self.time == self.time_limit - 1
        done = nca == self.m
        self.attacker_total_reward += au
        self.defender_total_reward += du
        if done:
            self.logger.info(
                f'Attacker/Defender: {self.attacker_total_reward / self.time_limit:.4}/{self.defender_total_reward / self.time_limit:.4}')

        # observation, reward, done, info
        return ({
                   'att': {
                       'action': action,
                       'last_reimage': self.last_reimage,
                       'success': success
                   },
                   'def': {
                       'action': self.defender_last_action,
                       'last_probe': self.last_probe
                   },
                   'time': self.time
               }, {
                   'att': au,
                   'def': du
               }, done, {})

    def reset(self):

        self.logger.info(
            f'Attacker/Defender: {self.attacker_total_reward / 1000:.4f}/{self.defender_total_reward / 1000:.4f}')

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

        self.attacker_total_reward = 0
        self.defender_total_reward = 0

        self.defender_last_action = -1

        self.defender = copy.deepcopy(self.defender_t)

        self.attacker_servers = []
        for i in range(self.m):
            self.attacker_servers.append({
                'status': -1,
                'progress': 0,
                'control': 0
            })

        return {
            'att': {
                'action': -1,
                'last_reimage': -1,
                'success': 0
            },
            'def': {
                'action': -1,
                'last_probe': -1
            },
            'time': 0
        }

    def render(self, mode='human'):
        self.logger.warning(f'LastProbe/LastReimage: {self.last_probe}/{self.last_reimage}')
        self.logger.warning(f'Server States: {self.servers}')
        time.sleep(5)
