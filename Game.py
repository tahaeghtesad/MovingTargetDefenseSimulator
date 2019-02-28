from enum import Enum
import random
import math


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

        self.time = 0


    def probe(self, server):
        if not 0 <= server < self.m:
            raise Exception('Chosen server is not in range')

        if self.servers[server]['status'] == -1:
            self.servers[server]['progress'] += 1

        if random.random() > (1 - math.exp(-self.success_increase_ratio * (self.servers[server]['progress'] + 1))): # 1 - e^alpha*(rho + 1)
            self.servers[server]['control'] = Party.Attacker

    def reimage(self, server):
        if not 0 <= server < self.m:
            raise Exception('Chosen server is not in range')

        if self.servers[server]['status'] != -1:
            return

        self.servers[server] = {
            'control': Party.Defender,
            'status': self.time,  # -1 means that it is up, a positive number is the time of re-image
            'progress': 0
        }
