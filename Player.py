import logging

class Player:
    def __init__(self):
        self.utility = 0
        self.last_util = 0

    def update_utility(self, u):
        logging.info(f'Received {u} utility')
        self.last_util = u
        self.utility += u
