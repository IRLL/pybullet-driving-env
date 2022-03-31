import gym
import numpy as np
import math
import pybullet as p
from pybullet_driving_env.resources.carreverse import CarReverse
from pybullet_driving_env.resources.plane import Plane
from pybullet_driving_env.resources.goal import Goal
from pybullet_driving_env.envs.simple_driving_env import SimpleDrivingEnv
from copy import deepcopy
import time
#import matplotlib.pyplot as plt


class DrivingGrid(SimpleDrivingEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = gym.spaces.box.Box(
            low=np.array([0, -.6, 0], dtype=np.float32),
            high=np.array([1, .6, 1], dtype=np.float32))

        """
            obs_space -- observation space for the environment
                car_qpos     -- x and y co-ordinates of the car, orientation of the car (Euler angles/pi), x and y velocities of the car
                segmentation -- A 3d occupancy map with Region occupied by the car along the first layer, that occupied by the obstacles along the second layer,
                                and the goal position along the third layer (not present for the agent setting the goal)  
        """
        obs_space = {
            'car_qpos':gym.spaces.box.Box(
                    low=np.array([-np.Inf, -np.Inf, -1, -1, -1, -np.Inf, -np.Inf], dtype=np.float32),
                    high=np.array([np.Inf,  np.Inf,  1,  1,  1,  np.Inf,  np.Inf], dtype=np.float32)
                ),
            
            'segmentation': gym.spaces.box.Box(
                    low = np.zeros((75,75,3)),
                    high = np.ones((75,75,3)),
            )
        }
        self.grid = np.zeros([21,21])
        self.observation_space = gym.spaces.dict.Dict(obs_space)
        self.np_random, _ = gym.utils.seeding.np_random()
        
        # use this if not rendering
        self.client = p.connect(p.GUI)

        #use this if rendering
        # self.client = p.connect(p.GUI)

        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)

        self.car = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.max_steps = 250
        self.steps = 0

        # initialize the obstacles and their dimentions
        self.pos_obstacles = None
        self.obstacle_dims = None
        self.initialize_obstacles()
        
        self.obstacle_mass = 1000
        self.max_half_length = 12

        # setting up the camera
        self.viewMatrix = p.computeViewMatrix(
                    cameraEyePosition=[0, 0, 25],
                    cameraTargetPosition=[0, 0, 0],
                    cameraUpVector=[0, 1, 0])
        self.projectionMatrix = p.computeProjectionMatrixFOV(
                    fov=50.0,
                    aspect=1.0,
                    nearVal=0.1,
                    farVal=100.5)


    def initialize_obstacles(self):
        # TODO: once in running condition, change the position initialization to np.zeros
        """
            Write your own initialization
            pos_onstacles changes every time reset for alice is called
        """
        self.grid[-1,:] = 1
        self.grid[0,:] = 1
        self.grid[:,0] = 1
        self.grid[:,-1] = 1

    def draw_obstacles(self):
        g2m,_ = self.grid2meter()
        box_half_height = 0.2
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i,j] == 1:
                    pos_obstacles = [
                                        -self.max_half_length + i*g2m,
                                        -self.max_half_length + j*g2m,
                                        0.1
                                    ]
                    visBoxId = p.createVisualShape(p.GEOM_BOX,
                                halfExtents=[g2m/2, g2m/2, box_half_height], rgbaColor=[0.62, 0.26, 0.67, 1])
                    colBoxId = p.createCollisionShape(p.GEOM_BOX,
                                halfExtents=[g2m/2 -0.05, g2m/2 -0.05, box_half_height])
                    p.createMultiBody(baseMass=self.obstacle_mass,
                        baseCollisionShapeIndex=colBoxId,
                        baseVisualShapeIndex=visBoxId,
                        basePosition=pos_obstacles)
    
    def grid2meter(self):
        return (2*self.max_half_length)/(self.grid.shape[0]), (2*self.max_half_length)/(self.grid.shape[1])

    def sample_spawn_pos(self):
        while True:
            i,j = np.random.randint((0,0), self.grid.shape, [2])
            if not self.grid[i,j] == 1:
                break
        g2m, _ = self.grid2meter()
        spawn = np.array([
                    -self.max_half_length + i*g2m,
                    -self.max_half_length + j*g2m,
                    0.2
                ])
        return spawn
    
    def sample_spawn_orientation(self):
        return np.random.uniform(-1,1, (4,))

    def reset_car(self, client, base_position, base_orientation):
        return CarReverse(client, base_position, base_orientation)

    def reset_obstacle_positions(self):
        """
            reset the obstacle positions based on some probability distribution
            Write your own reset function
        """
        self.grid[-1,:] = 1
        self.grid[0,:] = 1
        self.grid[:,0] = 1
        self.grid[:,-1] = 1
        self.grid[10,:] = 1
        self.grid[:,10] = 1
        self.grid[6:9, 10] = 0
        self.grid[13:16,10] = 0
        self.grid[10,6:9] = 0
        self.grid[10,13:16] = 0

    def close(self):
        p.disconnect(self.client)
