from enum import Enum
import random
import math
import logging
from tqdm import tqdm
from Attacker import BaseAttacker
from Defender import BaseDefender


class Party(Enum):
    Attacker = 1,
    Defender = 2


class Game:
    def __init__(self, m=10, downtime=7, success_increase_ratio=1, time_limit=400, probe_detection=0.):
        self.servers = [{
            'control': Party.Defender,
            'status': -1,  # -1 means that it is up, a positive number is the time of re-image
            'progress': 0
        }] * m

        self.m = m
        self.downtime = downtime
        self.success_increase_ratio = success_increase_ratio
        self.time_limit = time_limit
        self.probe_detection = probe_detection

        self.last_probe = -1  # -1 means that the last probe is not detectable
        self.last_reimage = -1  # -1 means that the last reimage is not detectable

        self.time = 0

    def probe(self, server):
        if not 0 <= server < self.m:
            raise Exception('Chosen server is not in range')

        if self.servers[server]['status'] == -1:
            self.servers[server]['progress'] += 1
            if random.random() > self.probe_detection:
                self.last_probe = server
            else:
                self.last_probe = -1

        if random.random() > (1 - math.exp(
                -self.success_increase_ratio * (self.servers[server]['progress'] + 1))):  # 1 - e^alpha*(rho + 1)
            self.servers[server]['control'] = Party.Attacker
            return True

        return False

    def reimage(self, server):
        if not 0 <= server < self.m:
            raise Exception('Chosen server is not in range')

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

    def utility(self, nc, nd, w=.3, tth_1=.5, tth_2=.5):
        return w * Game.sigmoid(nc / self.m, tth_1) + (1 - w) * Game.sigmoid((nc + nd) / self.m, tth_2)

    def play(self, attacker: BaseAttacker, defender: BaseDefender):
        ### Reseting state
        self.__init__(self.m, self.downtime, self.success_increase_ratio, self.time_limit, self.probe_detection)
        for self.time in tqdm(range(self.time_limit)):

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
            ncd = sum(server['control'] == Party.Defender for server in self.servers)
            nd = sum(server['status'] == -1 for server in self.servers)

            attacker.update_utility(self.utility(nca, nd))
            attacker.update_utility(self.utility(ncd, nd))
