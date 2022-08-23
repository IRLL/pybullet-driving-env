import gym
from ppo import PPO
from config import Config
import pybullet_driving_env
from pybullet_driving_env.envs.driving_grid import DrivingGrid
from math import prod


def main():
    env = gym.make("Hopper-v4")
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    config = Config()
    ppo = PPO(
        env=env, num_states=num_states, num_actions=num_actions, config=config
    )
    ppo.train()


def driving():
    env = gym.make("DrivingGrid-v0")
    num_states = prod(env.observation_space["car_qpos"].shape) + prod(
        env.observation_space["segmentation"].shape
    )
    num_actions = env.action_space.shape[0]
    config = Config()
    ppo = PPO(
        env=env, num_states=num_states, num_actions=num_actions, config=config
    )
    ppo.train()


if __name__ == "__main__":
    driving()
