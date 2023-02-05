import numpy as np
import pandas as pd
import gym
import minerl
import logging


def runOneEpisode():
    env = gym.make('CartPole-v1')
    done = False
    obs = env.reset()
    while not done:
        a = env.action_space.sample()
        # In BASALT environments, sending ESC action will end the episode
        # Lets not do that
        obs, r, done, _ = env.step(a)
        env.render()


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    # env = gym.make("MineRLBasaltFindCave-v0")
    # env = gym.make("MineRLBasaltBuildVillageHouse-v0")
    runOneEpisode()
