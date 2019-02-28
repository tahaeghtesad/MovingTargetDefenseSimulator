import random
import logging
from Player import Player


class BaseAttacker(Player):

    def __init__(self, m = 10, downtime = 7):
        super().__init__()
        self.m = m
        self.downtime = downtime

        self.servers = [{
            'status': -1,
            'progress': 0,
            'control': 0
        }]

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

        action = random.randint(0, self.m - 1)
        logging.warning(f'Doing random probe: {action}')

        success = probe(action)

        if success:
            self.servers[action]['control'] = 1

        logging.info(f'Probe was {"successful" if success else "unsuccessful"}.')
