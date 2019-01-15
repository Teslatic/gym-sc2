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

class Move2Beacon(Base):
    """
    Gym wrapper class for the pysc2 minigame "Move to Beacon".
    Inherits from GymSc2Base, which is a custom base class.
    """

    def __init__(self):
        super(Move2Beacon, self).__init__()

        self.map_name = 'MoveToBeacon'
        self.MAX_DISTANCE = np.sqrt(self.screen_dim_x**2
                                    + self.screen_dim_y**2)
        self.MIN_DISTANCE = 0

        # Action space info ?

    def reset(self):
        observation = super(Move2Beacon, self).environment_reset()
        self.beacon_center, self.marine_center, self.distance = \
            self.calc_distance(observation)

        return self.retrieve_step_info(observation)

    def setup(self, env_specs, mode="learning"):


        self.grid_dim_x = int(env_specs['GRID_DIM_X'])
        self.grid_dim_y = int(env_specs['GRID_DIM_Y'])

        self.action_fn = self.define_action_fn(env_specs['ACTION_TYPE'])
        self.reward_fn = self.define_reward_fn(env_specs['REWARD_TYPE'])

        return super(Move2Beacon, self).setup(env_specs, mode)






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
        elif reward_type == 'diff':
            return self.diff_reward_fn
        elif reward_type == 'distance':
            return self.distance_reward_fn
        else:
            raise("Specify reward function!")

        raise NotImplementedError

    def sparse_reward_fn(self, observation):
        """
        Sparse reward: If the marine hits a beacon reward=1, 0 else
        """
        return observation[0].reward

    def diff_reward_fn(self, observation):
        """
        Difference reward: If the marine hits a beacon reward=100,
        else it returns the covered distance.
        """
        reward_shaped = self.distance - self.distance_next

        if observation[0].reward == 1:
            reward_shaped = 100

        return reward_shaped

    def distance_reward_fn(self, observation):
        """
        Distance reward: If the marine hits a beacon reward=100,
        else it returns the absolute distance normalized on the max possible
        distance.
        """
        distance_reward = -1 * (self.distance - self.MIN_DISTANCE) \
            / (self.MAX_DISTANCE - self.MIN_DISTANCE).round(4)


        beacon_center, marine, distance = self.calc_distance(observation)




        if observation[0].reward == 1:
            return 10
        else:
            return distance_reward

    def calc_distance(self, observation):
        """
        Calculates the euclidean distance between beacon and marine.
        Using feature_screen.selected since marine vanishes behind beacon when
        using feature_screen.player_relative
        """
        actual_obs = observation[0]
        scrn_player = actual_obs.observation.feature_screen.player_relative
        scrn_select = actual_obs.observation.feature_screen.selected
        scrn_density = actual_obs.observation.feature_screen.unit_density

        state_added = scrn_select + scrn_density

        marine_center = np.mean(self.xy_locs(scrn_player == 1), axis=0).round()

        # first step
        if np.sum(scrn_select) == 0:
            marine_center = np.mean(self.xy_locs(scrn_player == 1), axis=0).round()
            # marine behind beacon
            if isinstance(marine_center, float):
                marine_center = np.mean(self.xy_locs(state_added == 2), axis=0).round()
        else:
            # normal navigation
            marine_center = np.mean(self.xy_locs(state_added == 2), axis=0).round()
            if isinstance(marine_center, float):
                marine_center = np.mean(self.xy_locs(state_added == 3), axis=0).round()

        beacon_center = np.mean(self.xy_locs(scrn_player == 3), axis=0).round()
        #
        # print(state_added)
        # print("---- Marine {} | {} Beacon ----".format(marine_center, beacon_center))
        # time.sleep(0.2)
        distance = math.hypot(beacon_center[0] - marine_center[0],
                              beacon_center[1] - marine_center[1])

        return beacon_center, marine_center, distance

    def retrieve_step_info(self, observation):
        """
        Extracts state and reward information from the pysc2 player relative
        layer and converts it into gym-like observation tuple.
        """

        beacon_next, marine_next, self.distance_next = \
            self.calc_distance(observation)

        obs, reward, done, info = super(Move2Beacon, self).retrieve_step_info(observation)

        self.distance = self.distance_next
        self.marine_center = marine_next
        self.beacon_center = beacon_next

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
