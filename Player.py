import logging


class Player:
    def __init__(self):
        self.logger = logging.getLogger(Player.__name__)
        self.utility = 0
        self.last_util = 0

    def update_utility(self, u):
        self.logger.debug(f'Received {u} utility')
        self.last_util = u
        self.utility += u

    def finalize(self):
        pass
