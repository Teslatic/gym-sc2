# gym imports
import gym
from gym.utils import seeding

# pysc2 imports
from pysc2.env import sc2_env
from pysc2.lib import actions, features

# python imports
import numpy as np
import math
import time


class GymSc2Env(gym.Env):
    """
    A wrapper class that uses the PYSC2 environment like a gym environment.
    """
    metadata = {'render.modes': ['human']}

    ###########################################################################
    # Initialization, custom setup and resetting
    ###########################################################################

    def __init__(self, env_file=None):
        """
        The constructordefines consts. for the Move2Beacon environment.
        call self.setup(env_specs) after creating the env. with gym.make()
        """
        self.finished = False
        self.cnt_episodes = 0
        self.map_name = 'MoveToBeacon'
        self.players = [sc2_env.Agent(sc2_env.Race.terran)]
        self.screen_dim_x = 84
        self.screen_dim_y = 64
        self.minimap_dim = 64
        self.game_steps = 0

        # TODO: initialize observation space and action space
        self.agent_interface = self.setup_interface()
        self.FIRST_STEP = 0
        self.LAST_STEP = 2
        self.MAX_DISTANCE = np.sqrt(self.screen_dim_x**2
                                    + self.screen_dim_y**2)
        self.MIN_DISTANCE = 0

    def setup(self, sc2_env_file, mode="learning"):
        """
        An additional setup function that allows some custom modifications of
        the environment after calling gym.make()

        From the official PYSC2 documentation:
        You must pass a resolution that you want to play at. You can send
        either feature layer resolution or rgb resolution or both. If you send
        both you must also choose which to use as your action space.
        Regardless of which you choose you must send both the screen and
        minimap resolutions.

        For each of the 4 resolutions, either specify size or both width and
        height. If you specify size then both width and height will take that
        value.

        Args:
            _only_use_kwargs: Don't pass args, only kwargs.
            map_name: Name of a SC2 map. Run bin/map_list to get the full list
                      of known maps. Alternatively, pass a Map instance. Take a
                      look at the docs in maps/README.md for more information
                      on available maps.
            players: A list of Agent and Bot instances that specify who will
                     play.
            agent_race: Deprecated. Use players instead.
            bot_race: Deprecated. Use players instead.
            difficulty: Deprecated. Use players instead.
            screen_size_px: Deprecated. Use agent_interface_formats instead.
            minimap_size_px: Deprecated. Use agent_interface_formats instead.
            agent_interface_format: A sequence containing one
                                    AgentInterfaceFormat per agent, matching
                                    the order of agents specified in the
                                    players list. Or a single
                                    AgentInterfaceFormat to be used for all
                                    agents.
            visualize: Whether to pop up a window showing the camera and
                        feature layers. This won't work without access to
                        a window manager.
            step_mul: How many game steps per agent step (action/observation).
                      None means use the map default.
            save_replay_episodes: Save a replay after this many episodes.
                                  Default of 0 means don't save replays.
            replay_dir: Directory to save replays. Required with
                        save_replay_episodes.
            game_steps_per_episode: Game steps per episode, independent of the
                                    step_mul. 0 means no limit. None means use
                                    the map default.
            random_seed: Random number seed to use when initializing the game.
            This lets you run repeatable games/tests.
        """
        # TODO NOT: Inconsistent check for Trur (Bool snd String)
        if mode == 'testing':
            visualize = True if sc2_env_file['TEST_VISUALIZE'] == 'True' else False
        else:
            visualize = True if sc2_env_file['VISUALIZE'] == True else False

        self.env = sc2_env.SC2Env(
            map_name=self.map_name,
            players=self.players,
            agent_interface_format=self.agent_interface,
            step_mul=int(sc2_env_file['STEP_MUL']),
            game_steps_per_episode=self.game_steps,
            visualize=visualize,
            # save_replay = sc2_env_file['SAVE_REPLAY'],
            # replay_dir = sc2_env_file['REPLAY_DIR'],
            )

        if mode == 'learning':
            self.episodes = int(sc2_env_file['EPISODES'])
        else:
            self.episodes = int(sc2_env_file['TEST_EPISODES'])

        self.step_mul = int(sc2_env_file['STEP_MUL'])

        self.grid_dim_x = int(sc2_env_file['GRID_DIM_X'])
        self.grid_dim_y = int(sc2_env_file['GRID_DIM_Y'])

        self.action_fn = self.define_action_fn(sc2_env_file['ACTION_TYPE'])
        self.reward_fn = self.define_reward_fn(sc2_env_file['REWARD_TYPE'])
        return self.reset()

    def setup_interface(self):
        """
        Setting up agent interface for the environment.
        """
        # TODO(vloeth): only screen dimension x is used because
        # of problems in case of using a tuple like (84, 64)
        agent_interface = features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=self.screen_dim_x,
                                                   minimap=self.minimap_dim),
            use_feature_units=True)
        return agent_interface

    def reset(self):
        """
        This method resets the environment and returns the initial state
        of the reset environment.
        """
        self.cnt_episodes += 1
        if self.cnt_episodes > self.episodes:
            print("Training/Testing finished.")
            self.env.close()
            self.finished = True
            exit()
        print("About to reset")
        observation = self.env.reset()

        self.beacon_center, self.marine_center, self.distance = \
            self.calc_distance(observation)
        return self.retrieve_step_info(observation)

    ###########################################################################
    # Define custom action function
    ###########################################################################

    def define_action_fn(self, action_type):
        """
        This method is used to define the action_fn which calculates the
        agent's action into a PYSC2-compatible action.

        The following action functions are available:
        1. Compass actions: The agent can select the compass actions:
            'left', 'up', 'right', 'down'
        2. Grid actions: The agent can select the index of a grid point:
            eg. 1,2,3 ... self.factor*self.factor
        3. Original actions: The agent can select the actions in
            the standard PYSC2 format: e.g. actions.FUNCTIONS.no_op()
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
        Output: an PYSC2 compatible action that moves the agent in the selected
        direction.
        """
        # print('action passed to compass action fn: {}'.format(action))

        if self.can_do(actions.FUNCTIONS.Move_screen.id):
            if action is 'left':
                if not (self.marine_center[0] <= 0):
                    return actions.FUNCTIONS.Move_screen("now",
                                                         self.marine_center +
                                                         (-self.step_mul, 0))
            if action is 'up':
                if not (self.marine_center[1] <= 0):
                    return actions.FUNCTIONS.Move_screen("now",
                                                         self.marine_center +
                                                         (0, -self.step_mul))
            if action is 'right':
                if not (self.marine_center[0] >= 83):
                    return actions.FUNCTIONS.Move_screen("now",
                                                         self.marine_center +
                                                         (self.step_mul, 0))
            if action is 'down':
                if not (self.marine_center[1] >= 63):
                    return actions.FUNCTIONS.Move_screen("now",
                                                         self.marine_center +
                                                         (0, self.step_mul))

        elif action == 'select_army' and self.can_do(
                                            actions.FUNCTIONS.select_army.id):
            return actions.FUNCTIONS.select_army("select")
        else:
            return actions.FUNCTIONS.no_op()

    def grid_action_fn(self, action):
        """
        Input: 1, 2, ... self.factor*self.factor
        Output: an PYSC2 compatible action that moves the agent to the
        selected grid point. For further information refer to the methods
        discretize_xy_grid
        """
        if self.can_do(actions.FUNCTIONS.Move_screen.id):
            action = actions.FUNCTIONS.Move_screen("now",
                                                   (self.xy_space[action][0],
                                                    self.xy_space[action][1]))
        elif self.can_do(actions.FUNCTIONS.select_army.id):
            action = actions.FUNCTIONS.select_army("select")
        else:
            action = actions.FUNCTIONS.no_op()

        return action

    def discretize_xy_grid(self):
        """
        discretize action coordinates in order to keep action space small
        """
        x_space = np.linspace(0, 83, self.grid_dim_x, dtype=int)
        y_space = np.linspace(0, 63, self.grid_dim_y, dtype=int)
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

    ###########################################################################
    # Performing a step
    ###########################################################################

    def step(self, action):
        """
        This function translates the action to a PYSC2-compatible format and
        performing the action on the environment.
        """
        pysc2_action = self.action_fn(action)
        # print('pysc2 action in gym step: {}'.format(pysc2_action))
        observation = self.env.step([pysc2_action])

        return self.retrieve_step_info(observation)

    def retrieve_step_info(self, observation):
        """
        Extracts state and reward information from the pysc2 player relative
        layer and converts it into gym-like observation tuple.
        """
        # state_height        = np.array(observation[0].observation.feature_screen.height_map)
        # state_visibility    = np.array(observation[0].observation.feature_screen.visibility_map)
        # state_creep         = np.array(observation[0].observation.feature_screen.creep)
        # state_power         = np.array(observation[0].observation.feature_screen.power)
        # state_pl_id         = np.array(observation[0].observation.feature_screen.player_id)
        state_pl_rel        = np.array(observation[0].observation.feature_screen.player_relative)
        # state_unit_type     = np.array(observation[0].observation.feature_screen.unit_type)
        state_selected      = np.array(observation[0].observation.feature_screen.selected)
        # state_unit_hp       = np.array(observation[0].observation.feature_screen.unit_hit_points)
        # state_unit_hp_ratio = np.array(observation[0].observation.feature_screen.unit_hit_points_ratio)
        # state_unit_en       = np.array(observation[0].observation.feature_screen.unit_energy)
        # state_unit_en_ratio = np.array(observation[0].observation.feature_screen.unit_energy_ratio)
        # state_unit_sh       = np.array(observation[0].observation.feature_screen.unit_shields)
        # state_unit_sh_ratio = np.array(observation[0].observation.feature_screen.unit_shields_ratio)
        state_density       = np.array(observation[0].observation.feature_screen.unit_density)
        # state_density_aa    = np.array(observation[0].observation.feature_screen.unit_density_aa)
        # state_effects       = np.array(observation[0].observation.feature_screen.effects)

        # state = np.stack([np.array(state_selected + state_density),
        #                 state_height,
        #                 state_visibility,
        #                 state_creep,
        #                 state_power,
        #                 state_pl_id,
        #                 state_pl_rel,
        #                 state_unit_type,
        #                 state_selected,
        #                 state_unit_hp,
        #                 state_unit_hp_ratio,
        #                 state_unit_en,
        #                 state_unit_en_ratio,
        #                 state_unit_sh,
        #                 state_unit_sh_ratio,
        #                 state_density,
        #                 state_density_aa])



        state = [np.array(state_selected + state_density)]

        beacon_next, marine_next, self.distance_next = \
            self.calc_distance(observation)
        reward = np.float32(self.reward_fn(observation))
        # self.dummy_reward = reward
        pysc2_score = observation[0].observation.score_cumulative[0]
        pysc2_reward = observation[0].reward
        if observation[0].step_type.value == self.LAST_STEP:
            last = True
        else:
            last = False
        # check if needed

        if observation[0].step_type.value == self.FIRST_STEP:
            first = True
        else:
            first = False

        done = last
        info = None
        self.available_actions = observation[0].observation.available_actions

        self.distance = self.distance_next
        self.marine_center = marine_next
        self.beacon_center = beacon_next

        obs = [state,
               first,
               last,
               self.distance,
               self.marine_center,
               self.beacon_center,
               pysc2_score,
               pysc2_reward]

        return obs, reward, done, info

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

    ###########################################################################
    # Gym compatible methods
    ###########################################################################

    def render(self, mode='human', close=False):
        """
        Not used since rendering is done by the PYSC2 API.
        """
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
