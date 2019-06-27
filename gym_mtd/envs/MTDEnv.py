import gym
from gym.spaces import *
import logging
from enum import Enum
import math
import random
import numpy as np
import time
import copy

from util.AttackerProcessor import AttackerProcessor
from util.DefenderProcessor import DefenderProcessor


class Party(Enum):
    Attacker = 1
    Defender = 2


class MovingTargetDefenceEnv(gym.Env):

    def __init__(self, m=10, downtime=7, alpha=.05, time_limit=1000, probe_detection=0., utenv=0, setting=0, ca=.2):
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
        self.episodes = 0

        self.attacker_servers = []
        for i in range(m):
            self.attacker_servers.append({
                'status': -1,
                'progress': 0,
                'control': 0
            })

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

        self.last_reimage = -1

        self.defender_last_action = server

        if server == -1:
            return

        if self.servers[server]['status'] != -1:
            return

        if self.servers[server]['control'] == Party.Attacker:
            self.last_reimage = server

        # Defender reimaged a server which attacker probed in last action: Don't tell defender that attacker even attacked!
        if self.last_probe == server:
            self.last_probe = -1

        self.servers[server] = {
            'control': Party.Defender,
            'status': self.time,  # -1 means that it is up, a positive number is the time of re-image
            'progress': 0
        }

    def step(self, action: tuple):  # Action[0] is the attacker's move, Action[1] the defender's.
        assert self.time < self.time_limit
        self.logger.debug(f'Round {self.time}/{self.time_limit}')
        ### Onlining servers

        for i in range(self.m):
            if self.servers[i]['status'] != -1:
                if self.servers[i]['status'] + self.downtime <= self.time:
                    self.servers[i]['status'] = -1

        ### Doing actions

        att_a, def_r = action

        success = self.probe(att_a)
        self.reimage(def_r)

        ### Calculate utility
        nca = sum(server['control'] == Party.Attacker for server in self.servers)
        ncd = sum(server['control'] == Party.Defender and server['status'] == -1 for server in self.servers)
        nd = sum(server['status'] > -1 for server in self.servers)

        assert nca + ncd + nd == self.m, "N_ca, N_cd, or N_d is calculated incorrectly!"

        au = self.utility(nca, nd, self.utenv[0], self.setting[0], self.setting[1]) + self.last_attack_cost
        du = self.utility(ncd, nd, self.utenv[1], self.setting[2], self.setting[3])

        self.time += 1

        self.logger.debug(f'Received {au} utility.')
        done = self.time == self.time_limit
        # done = nca == self.m
        self.attacker_total_reward += au
        self.defender_total_reward += du
        if done:
            self.logger.info(
                f'Game {self.episodes} - Attacker/Defender: {self.attacker_total_reward / self.time_limit:.4f}/{self.defender_total_reward / self.time_limit:.4f}')

        # observation, reward, done, info
        return ({
                   'att': {
                       'action': att_a,
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

        self.attacker_total_reward = 0
        self.defender_total_reward = 0

        self.defender_last_action = -1

        self.episodes += 1

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
        time.sleep(1)


class MTDAttackerEnv(MovingTargetDefenceEnv):

    def __init__(self, defender, m=10, downtime=7, alpha=.05, time_limit=1000, probe_detection=0., utenv=0, setting=0, ca=.2):
        super().__init__(m, downtime, alpha, time_limit, probe_detection, utenv, setting, ca)
        self.action_space = Discrete(m + 1)
        self.observation_space = MultiDiscrete([2, 7, 32, 2] * m)

        self.attacker_processor = AttackerProcessor(m, downtime)
        self.defender_processor = DefenderProcessor(m, downtime)

        self.defender_b = defender
        self.defender = copy.deepcopy(self.defender_b)

        self.last_defender_obs = None

    def step(self, action):

        attacker_action = self.attacker_processor.process_action(action)
        defender_action = self.defender.predict(self.last_defender_obs)
        observation, reward, done, info = super().step((attacker_action, defender_action))

        self.last_defender_obs, _, _, _ = self.defender_processor.process_step(observation, reward, done, info)

        return self.attacker_processor.process_step(observation, reward, done, info)

    def reset(self):
        self.defender = copy.deepcopy(self.defender_b)
        observation = super().reset()

        self.last_defender_obs = self.defender_processor.process_observation(observation)
        return self.attacker_processor.process_observation(observation)


class MTDDefenderEnv(MovingTargetDefenceEnv):

    def __init__(self, attacker, m=10, downtime=7, alpha=.05, time_limit=1000, probe_detection=0., utenv=0, setting=0, ca=.2):
        super().__init__(m, downtime, alpha, time_limit, probe_detection, utenv, setting, ca)

        self.action_space = Discrete(m + 1)
        self.observation_space = MultiDiscrete([2, 7, 32, 512, 512] * m)

        self.attacker_processor = AttackerProcessor(m, downtime)
        self.defender_processor = DefenderProcessor(m, downtime)

        self.attacker_b = attacker
        self.attacker = copy.deepcopy(self.attacker_b)

        self.last_attacker_obs = None

    def step(self, action):

        attacker_action = self.attacker.predict(self.last_attacker_obs)
        defender_action = self.defender_processor.process_action(action)
        observation, reward, done, info = super().step((attacker_action, defender_action))

        self.last_attacker_obs, _, _, _ = self.attacker_processor.process_step(observation, reward, done, info)

        return self.defender_processor.process_step(observation, reward, done, info)

    def reset(self):
        self.attacker = copy.deepcopy(self.attacker_b)
        observation = super().reset()

        self.last_attacker_obs = self.attacker_processor.process_observation(observation)
        return self.defender_processor.process_observation(observation)
