from gym.envs.registration import register

register(
    id='gym-sc2-m2b-v0',
    entry_point='gym_sc2.envs:Move2Beacon')
