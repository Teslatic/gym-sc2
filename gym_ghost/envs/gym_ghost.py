# gym imports
import gym
from gym import error, spaces, utils
from gym.utils import seeding

# pysc2 imports
from pysc2.env import sc2_env
from pysc2.lib import actions, features

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

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, mode='human', close=False):
        pass
