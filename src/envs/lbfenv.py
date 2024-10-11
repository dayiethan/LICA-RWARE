import lbforaging
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box
from gymnasium import Wrapper, spaces


class FlattenObservation(Wrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._flatten_obs(obs), info

    def step(self, actions):
        obs, rew, done, truncated, info = self.env.step(actions)
        return self._flatten_obs(obs), rew, done, truncated, info

    def _flatten_obs(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )

class LBFEnvWrapper(object):
    def __init__(self, env_name="Foraging-8x8-2p-1f-v3", seed=None):
        self.env = gym.make(env_name)
        self.n_agents = self.env.n_agents
        self.action_space = self.env.action_space
        self.observation_space = Box(low=0, high=1, shape=self.env.observation_space.shape, dtype=np.float32)
        self.episode_limit = 50  # Adjust based on environment specs
    
    def reset(self):
        obs = self.env.reset()
        return [obs[i] for i in range(self.n_agents)], self.get_state()
    
    def step(self, actions):
        obs, rewards, done, info = self.env.step(actions)
        terminated = done
        return [obs[i] for i in range(self.n_agents)], rewards, [terminated] * self.n_agents, {}

    def get_state(self):
        return [self.get_agent_state(self.env, i) for i in range(self.n_agents)]
    
    def get_agent_state(self, env, agent_idx):
        """
        Get the state of a specific agent by index.
        Returns the agent's position, level, and other relevant info.
        """
        agent = env.players[agent_idx]
        agent_position = agent.position
        agent_level = agent.level
        
        # Example: State could include global info as well as agent-specific info
        agent_state = {
            'agent_position': agent_position,
            'agent_level': agent_level,
            'grid_size': env.field_size,
            # 'food_positions': [food.position for food in env.food]
        }
    
        return agent_state

    def get_obs(self):
        # return [self.env.get_agent_obs(i) for i in range(self.n_agents)]
        obs = self.env.observation_space
        flat = FlattenObservation(self.env)
        return flat._flatten_obs(obs)

    def get_avail_actions(self):
        return [[1] * self.env.action_space[0].n for _ in range(self.n_agents)]
    
    def get_state_size(self):
        """Returns the size of the global state."""
        # Get grid size (e.g., 8x8) from the environment
        grid_size = self.env.field_size
        num_agents = self.env.n_agents
        num_food_items = self.env.max_num_food

        # State representation can include:
        # 1. Grid size: Flattened grid of shape (grid_width * grid_height)
        # 2. Agent information: For each agent, store position and level
        # 3. Food information: For each food item, store position

        # Flattened grid: Each cell can hold information about agents, food, or empty space
        grid_state_size = grid_size[0] * grid_size[1]

        # Agent information: Position (2D coordinates) and level (1 value per agent)
        agent_state_size = num_agents * (2 + 1)  # 2 for position (x, y), 1 for strength/level

        # Food information: Position (2D coordinates for each food item)
        food_state_size = num_food_items * 2  # Each food has a (x, y) position

        # Total state size is the sum of the grid, agent, and food state sizes
        total_state_size = grid_state_size + agent_state_size + food_state_size

        return total_state_size

    
    def get_env_info(self):
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.observation_space.shape,
            "n_actions": self.env.action_space[0].n,
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }
