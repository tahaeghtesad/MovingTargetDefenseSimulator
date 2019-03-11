from enum import Enum
import random
import math
import time
import logging
from tqdm import tqdm
import numpy as np
from BaseAttacker import BaseAttacker
from BaseDefender import BaseDefender


class Party(Enum):
    Attacker = 1
    Defender = 2

class Game:
    def __init__(self, m=10, downtime=7, alpha=.05, time_limit=1000, probe_detection=0., utenv=0, setting=0):
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

        self.last_probe = -1  # -1 means that the last probe is not detectable
        self.last_reimage = -1  # -1 means that the last reimage is not detectable

        self.utenv, self.setting = Game.get_params(utenv, setting)

        self.time = 0

    def probe(self, server):
        if not -1 <= server < self.m:
            raise Exception('Chosen server is not in range')

        if server == -1:
            return False

        if self.servers[server]['status'] == -1:
            self.servers[server]['progress'] += 1
            if random.random() >= self.probe_detection:
                self.last_probe = server
            else:
                self.last_probe = -1

            if random.random() < (1 - math.exp(
                    -self.alpha * (self.servers[server]['progress'] + 1))):  # 1 - e^-alpha*(rho + 1)
                self.servers[server]['control'] = Party.Attacker
                return True

        return False

    def reimage(self, server):
        if not -1 <= server < self.m:
            raise Exception('Chosen server is not in range')

        if server == -1:
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

    @staticmethod
    def sigmoid(x, tth, tsl=5):
        return 1. / (1. + math.exp(-tsl * (x - tth)))

    @staticmethod
    def get_params(env, setting): # env = [1: control/avail 2: control/config 3:disrupt/avail 4:disrupt/confid] - setting = [0:low 1:major 2:high]
        utenv = [(1, 1), (1, 0), (0, 1), (0, 0)]
        setenv = [(.2, .2, .2, .2), (.5, .5, .5, .5), (.8, .8, .8, .8)]
        return utenv[env], setenv[setting]

    def utility(self, nc, nd, w, tth_1, tth_2):
        return w * Game.sigmoid(nc / self.m, tth_1) + (1 - w) * Game.sigmoid((nc + nd) / self.m, tth_2)

    def play(self, attacker: BaseAttacker, defender: BaseDefender):
        for self.time in range(self.time_limit):
            self.logger.debug(f'Round {self.time}/{self.time_limit}')
            ### Onlining servers

            for i in range(self.m):
                if self.servers[i]['status'] != -1:
                    if self.servers[i]['status'] + self.downtime <= self.time:
                        self.servers[i]['status'] = -1

            ### Doing actions

            p_last_probe = self.last_probe
            p_last_reimage = self.last_reimage

            attacker.probe(self.time, p_last_reimage, self.probe)
            defender.reimage(self.time, p_last_probe, self.reimage)

            ### Calculate utility
            nca = sum(server['control'] == Party.Attacker for server in self.servers)
            ncd = sum(server['control'] == Party.Defender and server['status'] == -1 for server in self.servers)
            nd = sum(server['status'] > -1 for server in self.servers)

            attacker.update_utility(self.utility(nca, nd, self.utenv[0], self.setting[0], self.setting[1]))
            defender.update_utility(self.utility(ncd, nd, self.utenv[1], self.setting[2], self.setting[3]))
