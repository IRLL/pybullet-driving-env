import pybullet_driving_env
from pybullet_driving_env.envs.driving_grid import DrivingGrid
import numpy as np
import gym

# env = gym.make('SimpleDriving-v0')
env = gym.make('DrivingGrid-v0')
# env = DrivingGrid()
print(env.observation_space)
print(env.action_space)

for episode in range(5):
    for i in range(4):

        # set spawn position and orientation
        spawn_position = np.random.uniform(-10,10, (3,))
        spawn_orientation = np.random.uniform(-1,1, (4,))
        spawn_position[2] = 0.5
        
        obs = env.reset(12*np.ones((2,)),spawn_position, spawn_orientation)
        state = obs['car_qpos']; gridmap = obs['segmentation']
        
        done = False
        # run alice
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
        
        # set goal for bob and reset
        goal = obs['car_qpos'][:2]
        obs = env.reset(goal, spawn_position,spawn_orientation, agent = "bob")

        done = False
        #run bob
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action, 1.5, agent="bob")
        