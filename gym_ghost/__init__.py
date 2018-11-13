from gym.envs.registration import register

register(
    id='sc2-v0',
    entry_point='gym_ghost.envs:GymSc2Env',
)
