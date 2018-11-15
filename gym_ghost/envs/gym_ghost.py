# gym imports
import gym
from gym import error, spaces, utils
from gym.utils import seeding

# pysc2 imports
from pysc2.env import sc2_env
from pysc2.lib import actions, features

# python imports
import numpy as np
import math

from env_specs import mv2beacon_specs

class GymSc2Env(gym.Env):
    """
    A wrapper class that uses the PYSC2 environment like a gym environment.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, env_file = None):
        self.map_name = 'MoveToBeacon'
        self.players = [sc2_env.Agent(sc2_env.Race.terran)]
        self.screen_dim_x = 84
        self.screen_dim_y = 64
        self.minimap_dim = 64
        self.game_steps = 0
        # observation space und action space anlegen
        self.agent_interface = self.setup_interface()
        self.LAST_STEP = 2
        self.MAX_DISTANCE = np.sqrt(self.screen_dim_x**2 + self.screen_dim_y**2)
        self.MIN_DISTANCE = 0

    def define_action_fn(self, action_type):
        if action_type == 'compass':
            return self.compass_action_fn
        elif action_type == 'grid':
            self.xy_space = self.discretize_xy_grid()
            return self.grid_action_fn
        elif action_type == 'original':
            return self.original_action_fn
        else:
            raise("Specifiy action function!")

    def define_reward_fn(self, reward_type):
        if reward_type == 'diff':
            return self.diff_reward_fn
        elif reward_type == 'sparse':
            return self.sparse_reward_fn
        elif reward_type == 'distance':
            return self.distance_reward_fn
        else:
            raise("Specify reward function!")

    def discretize_xy_grid(self):
        """ "discretizing" action coordinates in order to keep action space small """
        x_space = np.linspace(0, 83, self.grid_factor, dtype = int)
        y_space = np.linspace(0, 63, self.grid_factor, dtype = int)
        xy_space = np.transpose([np.tile(x_space, len(y_space)),
                                   np.repeat(y_space, len(x_space))])

        return xy_space

    def grid_action_fn(self, action):
        if self.can_do(actions.FUNCTIONS.Move_screen.id):
            action = actions.FUNCTIONS.Move_screen("now",
                                        (self.xy_space[action][0], self.xy_space[action][1]))
        else:
            action = actions.FUNCTIONS.no_op()

        return action

    def can_do(self, action):
        """
        shortcut for checking if action is available at the moment
        """
        return action in self.available_actions 

    def compass_action_fn(self, action):
         if self.can_do(actions.FUNCTIONS.Move_screen.id):
            if action is 'left':
                if not (self.marine_center[0] <= 0):
                    return actions.FUNCTIONS.Move_screen("now", self.marine_center + (-self.step_mul, 0))
            if action is 'up':
                if not (self.marine_center[1] <= 0):
                    return actions.FUNCTIONS.Move_screen("now", self.marine_center + (0, -self.step_mul))
            if action is 'right':
                if not (self.marine_center[0] >= 83):
                    return actions.FUNCTIONS.Move_screen("now", self.marine_center + (self.step_mul, 0))
            if action is 'down':
                if not (self.marine_center[1] >= 63):
                    return actions.FUNCTIONS.Move_screen("now", self.marine_center + (0, self.step_mul))
         else:
            return actions.FUNCTIONS.no_op()


    def xy_locs(self, mask):
        y, x = mask.nonzero()

        return list(zip(x, y))

    def calc_distance(self, observation):
        """
        Calculates the euclidean distance between beacon and marine.
        Using feature_screen.selected since marine vanishes behind beacon when
        using feature_screen.player_relative
        """
        screen_player = observation[0].observation.feature_screen.player_relative
        screen_selected = observation[0].observation.feature_screen.selected

        marine_center = np.mean(self.xy_locs(screen_selected == 1), axis=0).round()
        beacon_center = np.mean(self.xy_locs(screen_player == 3), axis=0).round()
        if isinstance(marine_center, float):
            marine_center = beacon_center

        distance = math.hypot(beacon_center[0] - marine_center[0],
                              beacon_center[1] - marine_center[1])

        return beacon_center, marine_center, distance


    def sparse_reward_fn(self, observation):
        return observation[0].reward

    def diff_reward_fn(self, observation):
        reward_shaped = self.distance - self.distance_next
        if self.distance == 0.0:
            reward_shaped = 100
        return reward_shaped

    def distance_reward_fn(self, observation):
        scaling = lambda x : (x - self.MIN_DISTANCE)/(self.MAX_DISTANCE  - self.MIN_DISTANCE)
        distance_reward = -1 * scaling(self.distance).round(4)

        if observation[0].reward == 1:
            return 10
        else:
            return distance_reward

    def setup_interface(self):
        """
        Setting up agent interface for the environment.
        """
        # TODO(vloeth): only screen dimension x is used because of problems in case of using a tuple like (84, 64)
        agent_interface = features.AgentInterfaceFormat(
                            feature_dimensions=features.Dimensions(screen=self.screen_dim_x, minimap=self.minimap_dim),
                            use_feature_units=True)
        return agent_interface

    def setup(self, sc2_env_file):
        self.env = sc2_env.SC2Env(
            map_name=self.map_name,
            players=self.players,
            agent_interface_format=self.agent_interface,
            step_mul = sc2_env_file['STEP_MUL'],
            game_steps_per_episode=self.game_steps,
            visualize = sc2_env_file['VISUALIZE'],
            # save_replay = sc2_env_file['SAVE_REPLAY'],
            # replay_dir = sc2_env_file['REPLAY_DIR'],
            )

        self.step_mul = sc2_env_file['STEP_MUL']
        self.grid_factor = sc2_env_file['GRID_FACTOR']
        self.action_fn = self.define_action_fn(sc2_env_file['ACTION_TYPE'])
        self.reward_fn = self.define_reward_fn(sc2_env_file['REWARD_TYPE'])
        self.reset()

    def retrieve_step_info(self, observation):
        """
        Extracts state and reward information from the pysc2 player relative layer
        and converts it into gym-like observation tuple.
        """
        state = observation[0].observation.feature_screen.player_relative
        beacon_next, marine_next, self.distance_next = self.calc_distance(observation)
        reward = self.reward_fn(observation)
        done = True if observation[0].step_type.value == self.LAST_STEP else False
        # check if needed
        info = None

        self.available_actions = observation[0].observation.available_actions

        self.distance = self.distance_next
        self.marine_center = marine_next
        self.beacon_center = beacon_next

        return state, reward, done, info

    def step(self, action):
        pysc2_action = self.action_fn(action)

        observation = self.env.step([pysc2_action])

        return self.retrieve_step_info(observation)

    def reset(self):
        observation = self.env.reset()

        self.beacon_center, self.marine_center, self.distance = self.calc_distance(observation)

        return self.retrieve_step_info(observation)


    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]




