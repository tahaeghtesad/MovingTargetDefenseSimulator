import logging


class Player:
    def __init__(self, dr=.93):
        self.logger = logging.getLogger(Player.__name__)
        self.utility = 0
        self.last_util = 0
        self.dr = dr

        self.timestep = 0

    def set_dr(self, dr):
        self.dr = dr

    def update_utility(self, u):
        self.logger.debug(f'Received {u} utility')
        self.last_util = u
        # self.utility = self.utility + u * self.dr ** self.timestep
        self.utility += u
        self.timestep += 1

    def finalize(self, b):
        pass
