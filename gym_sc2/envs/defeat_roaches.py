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

class DefeatRoaches(Base):
    """
    Gym wrapper class for the pysc2 minigame "Defeat roaches".
    Inherits from Base, which is a custom base class.
    """

    def __init__(self):
        super(DefeatRoaches, self).__init__()

        self.map_name = 'DefeatRoaches'
        self.distance = None
        self.marine_center = None
        self.beacon_center = None

        # Action space info ?

    def reset(self):
        observation = super(DefeatRoaches, self).environment_reset()

        return self.retrieve_step_info(observation)

    def setup(self, env_specs, mode="learning"):
        self.grid_dim_x = int(env_specs['GRID_DIM_X'])
        self.grid_dim_y = int(env_specs['GRID_DIM_Y'])

        self.action_fn = self.define_action_fn(env_specs['ACTION_TYPE'])
        self.reward_fn = self.define_reward_fn(env_specs['REWARD_TYPE'])

        return super(DefeatRoaches, self).setup(env_specs, mode)

    ###########################################################################
    # Define reward function
    ###########################################################################

    def define_reward_fn(self, reward_type):
        """
        This method is used to define the reward_fn which calculates the
        agent's reward.

        The following reward functions are available:
        1. Sparse reward: If the marine hits a beacon reward=1, 0 else
        2. Diff reward: If the marine hits a beacon reward=100,
            else it returns the covered distance.
        3. Distance reward: If the marine hits a beacon reward=100,
            else it returns the absolute distance normalized on the
            max possible distance.
        """
        if reward_type == 'sparse':
            return self.sparse_reward_fn

        raise NotImplementedError

    def sparse_reward_fn(self, observation):
        """
        Sparse reward: If the marine hits a beacon reward=1, 0 else
        """
        return observation[0].reward

    def grid_action_fn(self, action):
        """
        Input: 1, 2, ... self.factor*self.factor
        Output: an PYSC2 compatible action that moves the agent to the
        selected grid point. For further information refer to the methods
        discretize_xy_grid
        """
        if self.can_do(actions.FUNCTIONS.Attack_screen.id):
            action = actions.FUNCTIONS.Attack_screen("now",
                                                   (self.xy_space[action][0],
                                                    self.xy_space[action][1]))
        elif self.can_do(actions.FUNCTIONS.select_army.id):
            action = actions.FUNCTIONS.select_army("select")
        else:
            action = actions.FUNCTIONS.no_op()

        return action

    def retrieve_step_info(self, observation):
        """
        Extracts state and reward information from the pysc2 player relative
        layer and converts it into gym-like observation tuple.
        """

        obs, reward, done, info = super(DefeatRoaches, self).retrieve_step_info(observation)


        STATE = 0
        FIRST = 1
        LAST = 2
        PYSC2_SCORE = 3
        PYSC2_REWARD = 4

        obs_mv2beacon = [obs[STATE],
               obs[FIRST],
               obs[LAST],
               self.distance,
               self.marine_center,
               self.beacon_center,
               obs[PYSC2_SCORE],
               obs[PYSC2_REWARD]]

        return obs_mv2beacon, reward, done, info
