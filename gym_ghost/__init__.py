#from gym.envs.gym_ghost.gym_ghost import GymSc2Env
from gym.envs.registration import register

register(
    id='sc2-v0',
    entry_point='gym_ghost.envs:GymSc2Env',
)
