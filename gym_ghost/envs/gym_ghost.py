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

class GymSc2Env(gym.Env):
    """
    A wrapper class that uses the PYSC2 environment like a gym environment.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.visualize = True
        self.map_name = 'MoveToBeacon'
        self.players = [sc2_env.Agent(sc2_env.Race.terran)]
        self.screen_dim = 84
        self.minimap_dim = 64
        self.step_mul = 4
        self.game_steps = 0
        self.save_replay = False,
        self.replay_dir = None
        # observation space und action space anlegen
        self.agent_interface = self.setup_interface()
        self.env = self.setup()
        # first reset to get distance initialization of marine - beacon
        self.LAST_STEP = 2

        self.factor = 10

        self.reward_type = 'shaped'

        self.action_type = 'grid'

        if self.action_type == 'compass':
            self.action_fn = self.compass_action_fn
        elif self.action_type == 'grid':
            self.xy_space = self.discretize_xy_grid()
            self.action_fn = self.grid_action_fn
        elif self.action_type == 'original':
            self.action_fn = self.original_action_fn
        else:
            raise("Specifiy action function!")

        if self.reward_type == 'shaped':
            self.reward_fn = self.shaped_reward_fn
        elif self.reward_type == 'sparse':
            self.reward_fn = self.sparse_reward_fn
        else:
            raise("Specify reward function!")
        self.reset()

    def discretize_xy_grid(self):
        """ "discretizing" action coordinates in order to keep action space small """
        x_space = np.linspace(0, 83, self.factor, dtype = int)
        y_space = np.linspace(0, 63, self.factor, dtype = int)
        xy_space = np.transpose([np.tile(x_space, len(y_space)),
                                   np.repeat(y_space, len(x_space))])

        return xy_space

    def grid_action_fn(self, action):
        print(self.xy_space, self.xy_space.shape)
        print()
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

    def shaped_reward_fn(self, observation):

        beacon_next, marine_next, distance_next = self.calc_distance(observation)


        reward_shaped = self.distance - distance_next
        if self.distance == 0.0:
            reward_shaped = 100

        self.distance = distance_next
        self.marine_center = marine_next
        self.beacon_center = beacon_next
        return reward_shaped


    def setup_interface(self):
        """
        Setting up agent interface for the environment.
        """
        agent_interface = features.AgentInterfaceFormat(
                            feature_dimensions=features.Dimensions(screen=self.screen_dim, minimap=self.minimap_dim),
                            use_feature_units=True)
        return agent_interface

    def setup(self):
        env = sc2_env.SC2Env(
            map_name=self.map_name,
            players=self.players,
            agent_interface_format=self.agent_interface,
            step_mul=self.step_mul,
            game_steps_per_episode=self.game_steps,
            visualize=self.visualize)
            # save_replay_episodes=self.save_replay
            # replay_dir=self.replay_dir)

        return env

    def retrieve_step_info(self, observation):
        """
        Extracts state and reward information from the pysc2 player relative layer
        and converts it into gym-like observation tuple.
        """
        state = observation[0].observation.feature_screen.player_relative
        reward = self.reward_fn(observation)
        done = True if observation[0].step_type.value == self.LAST_STEP else False
        # check if needed
        info = None

        self.available_actions = observation[0].observation.available_actions
        print(self.available_actions)

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
