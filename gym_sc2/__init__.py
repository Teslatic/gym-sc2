from gym.envs.registration import register

register(
    id='gym-sc2-m2b-v0',
    entry_point='gym_sc2.envs:Move2Beacon')

register(
    id='gym-sc2-mineralshards-v0',
    entry_point='gym_sc2.envs:CollectMineralShards')

register(
    id='gym-sc2-defeatroaches-v0',
    entry_point='gym_sc2.envs:DefeatRoaches')
