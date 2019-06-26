import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='MTD-v0',
    entry_point='gym_mtd.envs:MovingTargetDefenceEnv',
)

register(
    id='MTDAtt-v0',
    entry_point='gym_mtd.envs:MTDAttackerEnv'
)
