import matplotlib.pyplot as plt
import numpy as np
import lbforaging
import gymnasium as gym
import time


# TEMPLATE: "Foraging-{GRID_SIZE}x{GRID_SIZE}-{PLAYER COUNT}p-{FOOD LOCATIONS}f{-coop IF COOPERATIVE MODE}-v0"
env = gym.make("Foraging-8x8-2p-2f-c")
env.reset()

for t in range(100):
    actions = env.action_space.sample()
    nobs, nreward, ndone, ninfo, test = env.step(actions)
    print("nobs:", nobs)
    print("nreward:", nreward)
    # print("ndone:", ndone)
    # print("ninfo:", ninfo)
    print("actions:", actions)

    env.render()
    time.sleep(0.1)

    if np.array(ndone).all():
        break

env.close()