from functools import partial
# from smac.env import MultiAgentEnv, StarCraft2Env
from envs.lbfenv import LBFEnvWrapper
from envs.multiagentenv import MultiAgentEnv
from envs.gymma import GymmaWrapper
import sys
import os

# def env_fn(env, **kwargs):
#     print(f' Kwargs: {kwargs}' )
#     if isinstance(env, str):
#         # Check for the specific environment type and return the correct callable
#         if env == "lbf":
#             return GymmaWrapper(**kwargs)  # Create the environment with any additional parameters
#         else:
#             raise ValueError(f"Unknown environment: {env}")
#     return env(**kwargs)

def gymma_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return GymmaWrapper(**kwargs)

REGISTRY = {}
REGISTRY["gymma"] = gymma_fn

# if sys.platform == "linux":
#     os.environ.setdefault("SC2PATH",
#                           os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
