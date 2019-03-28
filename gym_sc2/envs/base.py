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

class Base(gym.Env):
    """
    A wrapper class that uses the PYSC2 environment like a gym environment.
    It is the basic class from which all specific minigame environments should
    inherit.
    """
    ###########################################################################
    # Initialization, custom setup and resetting
    ###########################################################################

    def __init__(self):
        """
        The constructordefines consts. for the Move2Beacon environment.
        call self.setup(env_specs) after creating the env. with gym.make()
        """
        self.finished = False
        self.cnt_episodes = 0
        self.players = [sc2_env.Agent(sc2_env.Race.terran)]
        self.screen_dim_x = 84
        self.screen_dim_y = 64
        self.minimap_dim = 64
        self.game_steps = 0


        # TODO: initialize observation space and action space
        self.agent_interface = self.setup_interface()
        self.FIRST_STEP = 0
        self.LAST_STEP = 2

    def setup(self, env_specs, mode="learning"):
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
        # TODO NOT: Inconsistent check for True (Bool snd String)

        # environment file cant be passed to gym make, hence the extra setup
        # function.
        self.step_mul = int(env_specs['STEP_MUL'])
        # self.mode = env_specs['MODE']
        self.mode = mode
        print(self.mode)
        if self.mode == 'testing':
            self.visualize = True if (env_specs['TEST_VISUALIZE'] == "True") else False
            self.episodes = int(env_specs['TEST_EPISODES'])
        elif self.mode == 'learning':
            self.visualize = True if (env_specs['VISUALIZE'] == "True") else False
            self.episodes = int(env_specs['EPISODES'])
        else:
            print("Current mode not known.")
            exit()
        self.env = sc2_env.SC2Env(
            map_name=self.map_name,
            players=self.players,
            agent_interface_format=self.agent_interface,
            step_mul=self.step_mul,
            game_steps_per_episode=self.game_steps,
            visualize=self.visualize,
            # save_replay = sc2_env_file['SAVE_REPLAY'],
            # replay_dir = sc2_env_file['REPLAY_DIR'],
            )

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

    def environment_reset(self):
        """
        Actual environment reset method.
        """
        self.cnt_episodes += 1
        if self.cnt_episodes > self.episodes:
            print("Training/Testing finished.")
            self.env.close()
            self.finished = True
            exit()
        print("Environment initialized. Fresh episode starting.")
        return self.env.reset()

    def reset(self):
        """
        This method calls the environment reset method and returns initial state
        of the reset environment.
        """
        observation = self.environment_reset()
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
        3. Pysc2 actions: The agent can select the actions in
            the standard PYSC2 format: e.g. actions.FUNCTIONS.no_op()
        """
        if action_type == 'compass':
            return self.compass_action_fn
        elif action_type == 'grid':
            self.xy_space = self.discretize_xy_grid()
            return self.grid_action_fn
        elif action_type == 'pysc2':
            return self.pysc2_action_fn
        elif action_type == 'minigame':
            return self.minigame_action_fn
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
                    target_location = self.marine_center + (-self.step_mul, 0)
            elif action is 'up':
                if not (self.marine_center[1] <= 0):
                    target_location = self.marine_center + (0, -self.step_mul)
            elif action is 'right':
                if not (self.marine_center[0] >= 83):
                    target_location = self.marine_center + (self.step_mul, 0)
            elif action is 'down':
                if not (self.marine_center[1] >= 63):
                    target_location = self.marine_center + (0, self.step_mul)
            else:
                print("Action is not known for compass action space!")
                exit()

            target_location = self.check_target_location(target_location)
            return actions.FUNCTIONS.Move_screen("now", target_location)

        elif self.can_do(actions.FUNCTIONS.select_army.id):
            return actions.FUNCTIONS.select_army("select")
        else:
            return actions.FUNCTIONS.no_op()

    def check_target_location(self, target_location):
        """
        Selects a target position within the boundaries.
        """
        if (target_location[0] < 0):
            target_location[0] = 0
        elif (target_location[0] > 83):
            target_location[0] = 83
        if (target_location[1] < 0):
            target_location[1] = 0
        elif (target_location[1] > 63):
            target_location[1] = 63
        return target_location

    def minigame_action_fn(self, action):
        """
        Input: PYSC2 action index
        Output: A PYSC2 compatible action (in the minigame action space).
        """
        if not self.can_do(action.id):
            print("Could not perform action. Performing no_op instead.")
            return actions.FUNCTIONS.no_op()
        elif int(action.id) == 0:
            return actions.FUNCTIONS.no_op()
        elif int(action.id) == 7:
            return actions.FUNCTIONS.select_army("select")
        elif int(action.id) == 331:
            return actions.FUNCTIONS.Move_screen("now", (14, 35))
        else:
            print("Action is not available in the current action space.")
            return actions.FUNCTIONS.no_op()

    def pysc2_action_fn(self, action):
        """
        Input: PYSC2 action index
        Output: A PYSC2 compatible action.
        """
        if not self.can_do(action.id):
            print("Could not perform action. Performing no_op instead.")
            return actions.FUNCTIONS.no_op()
        elif int(action.id) == 0:
            return actions.FUNCTIONS.no_op()
        elif int(action.id) == 7:
            return actions.FUNCTIONS.select_army("select")
        elif int(action.id) == 331:
            return actions.FUNCTIONS.Move_screen("now", (14, 35))
        else:
            print("Action is not available in the current action space.")
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
    # Performing a step
    ###########################################################################

    def step(self, action):
        """
        This function translates the action to a PYSC2-compatible format and
        performing the action on the environment.
        """
        pysc2_action = self.action_fn(action)
        # print("Action sent to env: " + str(action))
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
        reward = np.float32(self.reward_fn(observation))

        pysc2_score = observation[0].observation.score_cumulative[0]
        pysc2_reward = observation[0].reward

        if observation[0].step_type.value == self.LAST_STEP:
            last = True
        else:
            last = False

        if observation[0].step_type.value == self.FIRST_STEP:
            first = True
        else:
            first = False

        done = last
        info = None
        self.available_actions = observation[0].observation.available_actions

        obs = [state,
               first,
               last,
               pysc2_score,
               pysc2_reward,
               self.available_actions]

        return obs, reward, done, info

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
