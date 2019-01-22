# gym imports
import gym
from gym_sc2.envs.base import Base
from gym.utils import seeding

# pysc2 imports
from pysc2.env import sc2_env
from pysc2.lib import actions, features

# python imports
import numpy as np
import math
import time

class CollectMineralShards(Base):
    """
    Gym wrapper class for the pysc2 minigame "Collect Mineral Shards".
    Inherits from GymSc2Base, which is a custom base class.
    """

    def __init__(self):
        super(CollectMineralShards, self).__init__()

        self.map_name = 'CollectMineralShards'
        self.distance = None
        self.marine_center = None
        self.beacon_center = None

    def reset(self):
        observation = super(CollectMineralShards, self).environment_reset()

        return self.retrieve_step_info(observation)


    def setup(self, env_specs, mode="learning"):


        self.grid_dim_x = int(env_specs['GRID_DIM_X'])
        self.grid_dim_y = int(env_specs['GRID_DIM_Y'])

        self.action_fn = self.define_action_fn(env_specs['ACTION_TYPE'])
        self.reward_fn = self.define_reward_fn(env_specs['REWARD_TYPE'])

        return super(CollectMineralShards, self).setup(env_specs, mode)


    def define_reward_fn(self, reward_type):
        """
        This method is used to define the reward_fn which calculates the
        agent's reward.

        The following reward functions are available:
        1. sparse reward: +1 for each collected shard
        """
        if reward_type == 'sparse':
            return self.sparse_reward_fn
        else:
            raise("Specify reward function!")


    def sparse_reward_fn(self, observation):
        """
        sparse reward: +1 for each collected shard
        """
        return observation[0].reward



    def retrieve_step_info(self, observation):
        """
        Extracts state and reward information from the pysc2 player relative
        layer and converts it into gym-like observation tuple.
        """

        obs, reward, done, info = super(CollectMineralShards, self).retrieve_step_info(observation)

        STATE = 0
        FIRST = 1
        LAST = 2
        PYSC2_SCORE = 3
        PYSC2_REWARD = 4

        obs_collect_mineral_shards =    [obs[STATE],
                                        obs[FIRST],
                                        obs[LAST],
                                        self.distance,
                                        self.marine_center,
                                        self.beacon_center,
                                        obs[PYSC2_SCORE],
                                        obs[PYSC2_REWARD]]

        return obs_collect_mineral_shards, reward, done, info
