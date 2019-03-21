import random
import logging
from Player import Player


class BaseAttacker(Player):

    def __init__(self, m=10, downtime=7):
        super().__init__()
        self.logger = logging.getLogger(BaseAttacker.__name__)
        self.m = m
        self.downtime = downtime

        self.servers = []
        for i in range(m):
            self.servers.append({
                'status': -1,
                'progress': 0,
                'control': 0
            })

    def probe(self, time, last_reimage, probe):
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

        ### Choosing action
        action = self.select_action(time)
        self.logger.debug(f'Selecting server {action} to probe.')

        success = probe(action)

        if action == -1:
            return

        if success == 1:
            self.servers[action]['control'] = 1
            self.servers[action]['progress'] += 1
            self.servers[action]['status'] = -1
            self.logger.debug('Probe was successful.')
        elif success == 0:
            self.servers[action]['progress'] += 1
            self.servers[action]['control'] = 0
            self.servers[action]['status'] = -1
            self.logger.debug('Probe was unsuccessful.')
        elif success == -1:
            self.servers[action]['status'] = self.servers[action]['status'] if self.servers[action]['status'] != -1 else time
            self.servers[action]['progress'] = 0
            self.servers[action]['control'] = 0
            self.logger.debug('Server was down.')

    def select_action(self, time):
        # action = random.randint(-1, self.m - 1)
        action = -1
        # self.logger.warning(f'Doing random probe: {action}')
        return action
