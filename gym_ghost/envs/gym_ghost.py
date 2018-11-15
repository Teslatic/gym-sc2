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

    ##################################################################################
    # Initialization, custom setup and resetting
    ##################################################################################

    def __init__(self, env_file = None):
        """
        This constructor method defines constants for the Move2Beacon environment.
        Do not forget to call self.setup(env_specs) after creating the environment with gym.make()
        """
        self.map_name = 'MoveToBeacon'
        self.players = [sc2_env.Agent(sc2_env.Race.terran)]
        self.screen_dim_x = 84
        self.screen_dim_y = 64
        self.minimap_dim = 64
        self.game_steps = 0
        # TODO: initialize observation space and action space
        self.agent_interface = self.setup_interface()
        self.LAST_STEP = 2
        self.MAX_DISTANCE = np.sqrt(self.screen_dim_x**2 + self.screen_dim_y**2)
        self.MIN_DISTANCE = 0

    def setup(self, sc2_env_file):
        """
        An additional setup function that allows some custom modifications of the environment after
        calling gym.make()
        """
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

    def setup_interface(self):
        """
        Setting up agent interface for the environment.
        """
        # TODO(vloeth): only screen dimension x is used because of problems in case of using a tuple like (84, 64)
        agent_interface = features.AgentInterfaceFormat(
                            feature_dimensions=features.Dimensions(screen=self.screen_dim_x, minimap=self.minimap_dim),
                            use_feature_units=True)
        return agent_interface

    def reset(self):
        """
        This method resets the environment and returns the initial state of the reset environment.
        """
        observation = self.env.reset()

        self.beacon_center, self.marine_center, self.distance = self.calc_distance(observation)

        return self.retrieve_step_info(observation)

    ##################################################################################
    # Define custom action function
    ##################################################################################

    def define_action_fn(self, action_type):
        """
        This method is used to define the action_fn which calculates the agent's action into a 
        PYSC2-compatible action.

        The following action functions are available:
        1. Compass actions: The agent can select the compass actions: 'left', 'up', 'right', 'down'
        2. Grid actions: The agent can select the index of a grid point: eg. 1,2,3 ... self.factor*self.factor
        3. Original actions: The agent can select the actions in the standard PYSC2 format: e.g. actions.FUNCTIONS.no_op()
        """
        if action_type == 'compass':
            return self.compass_action_fn
        elif action_type == 'grid':
            self.xy_space = self.discretize_xy_grid()
            return self.grid_action_fn
        elif action_type == 'original':
            return self.original_action_fn
        else:
            raise("Specifiy action function!")

    def compass_action_fn(self, action):
        """
        Input: 'left', 'up', 'right', 'down'
        Output: an PYSC2 compatible action that moves the agent in the selected direction.
        """
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

    def grid_action_fn(self, action):
        """
        Input: 1, 2, ... self.factor*self.factor
        Output: an PYSC2 compatible action that moves the agent to the selected grid point.
        For further information refer to the methods discretize_xy_grid
        """
        if self.can_do(actions.FUNCTIONS.Move_screen.id):
            action = actions.FUNCTIONS.Move_screen("now",
                                        (self.xy_space[action][0], self.xy_space[action][1]))
        else:
            action = actions.FUNCTIONS.no_op()

        return action

    def discretize_xy_grid(self):
        """ "discretizing" action coordinates in order to keep action space small """
        x_space = np.linspace(0, 83, self.grid_factor, dtype = int)
        y_space = np.linspace(0, 63, self.grid_factor, dtype = int)
        xy_space = np.transpose([np.tile(x_space, len(y_space)),
                                   np.repeat(y_space, len(x_space))])

        return xy_space

    def can_do(self, action):
        """
        shortcut for checking if action is available at the moment
        """
        return action in self.available_actions 

    def xy_locs(self, mask):
        """
        TODO: Proper description!
        """
        y, x = mask.nonzero()

        return list(zip(x, y))


    ##################################################################################
    # Define reward function
    ##################################################################################

    def define_reward_fn(self, reward_type):
        """
        This method is used to define the reward_fn which calculates the agent's reward.

        The following reward functions are available:
        1. Sparse reward: If the marine hits a beacon reward=1, 0 else
        2. Diff reward: If the marine hits a beacon reward=100, else it returns the covered distance.
        3. Distance reward: If the marine hits a beacon reward=100, else it returns the absolute distance normalized on the max possible distance.
        """
        if reward_type == 'sparse':
            return self.sparse_reward_fn
        elif reward_type == 'diff':
            return self.diff_reward_fn
        elif reward_type == 'distance':
            return self.distance_reward_fn
        else:
            raise("Specify reward function!")

    def sparse_reward_fn(self, observation):
        """
        Sparse reward: If the marine hits a beacon reward=1, 0 else
        """
        return observation[0].reward

    def diff_reward_fn(self, observation):
        """
        Difference reward: If the marine hits a beacon reward=100, else it returns the covered distance.
        """
        reward_shaped = self.distance - self.distance_next
        if self.distance == 0.0:
            reward_shaped = 100
        return reward_shaped

    def distance_reward_fn(self, observation):
        """
        Distance reward: If the marine hits a beacon reward=100, else it returns the absolute distance normalized on the max possible distance.
        """
        scaling = lambda x : (x - self.MIN_DISTANCE)/(self.MAX_DISTANCE  - self.MIN_DISTANCE)
        distance_reward = -1 * scaling(self.distance).round(4)

        if observation[0].reward == 1:
            return 10
        else:
            return distance_reward

    ##################################################################################
    # Performing a step
    ##################################################################################

    def step(self, action):
        """
        The step function of the agent is translating the action to a PYSC2-compatible
        format and performing the action on the environment.
        """
        pysc2_action = self.action_fn(action)
        observation = self.env.step([pysc2_action])

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



        return self.retrieve_step_info(observation)


    ##################################################################################
    # Gym compatible methods
    ##################################################################################

    def render(self, mode='human', close=False):
        """
        Not used since rendering is done by the PYSC2 API.
        """
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]




