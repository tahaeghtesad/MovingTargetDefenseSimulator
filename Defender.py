import logging
import random
from Player import Player


class BaseDefender(Player):

    def __init__(self, m=10, downtime=7):
        super().__init__()
        self.logger = logging.getLogger(BaseDefender.__name__)
        self.m = m
        self.downtime = downtime

        self.servers = []
        for i in range(m):
            self.servers.append({
                'status': -1,
                'progress': 0
            })

    def reimage(self, time, last_probe, reimage):

        ### Updating state

        for s in self.servers:
            if s['status'] != -1:
                if s['status'] + self.downtime <= time:
                    s['status'] = -1

        if not -1 <= last_probe < self.m:
            raise Exception('Out of range server.')

        if last_probe != -1:
            self.servers[last_probe]['progress'] += 1

        ### Performing

        # action = random.randint(-1, self.m - 1)
        action = -1
        self.logger.warning(f'Doing random reimage: {action}')
        reimage(action)

        if self.servers[action]['status'] == -1:
            self.logger.info('Reimage was successful.')
            self.servers[action]['status'] = time
        else:
            self.logger.info('Reimage was unsuccessful.')
