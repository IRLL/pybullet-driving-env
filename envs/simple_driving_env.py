import gym
import numpy as np
import math
import pybullet as p
from resources.car import Car
from resources.plane import Plane
from resources.goal import Goal
from copy import deepcopy
import time
#import matplotlib.pyplot as plt


class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = gym.spaces.box.Box(
            low=np.array([0, -.6], dtype=np.float32),
            high=np.array([1, .6], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-10, -10, -np.pi, -np.pi, -np.pi, -5, -5], dtype=np.float32),
            high=np.array([10, 10, np.pi, np.pi, np.pi, 5, 5], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.DIRECT)
        # print("Done")
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
        # self.numobstacles = 6
        # self.pos_obstacles = np.array([[7,7,0.1],
        #                     [-3,4,0.1],
        #                     [5,-6,0.1],
        #                     [-7,-5,0.1],
        #                     [1,-2, 0.1],
        #                     [0,12,0.1],
        #                     [0,-12,0.1],
        #                     [12,0,0.1],
        #                     [-12,0,0.1]])
        self.pos_obstacles = np.array([[7,7,0.1],
                            [-3,4,0.1],
                            [5,-6,0.1],
                            [-7,-5,0.1],
                            [1,-2, 0.1],
                            [0,12,0.1],
                            [0,-12,0.1],
                            [12,0,0.1],
                            [-12,0,0.1]])

        wall_width = 11
        self.obstacle_dims = np.array([[1,1,0.2],
                            [1,1,0.2],
                            [1,1,0.2],
                            [1,1,0.2],
                            [1,1,0.2],
                            [wall_width,1,0.2],
                            [wall_width,1,0.2],
                            [1,wall_width,0.2],
                            [1,wall_width,0.2]])
        # boxHalfLength = 5
        # boxHalfWidth = 5
        # boxHalfHeight = 5

        # self.colBoxId = p.createCollisionShape(p.GEOM_BOX,
        #                           halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight])
        self.obstacle_mass = 1000
        self.viewMatrix = p.computeViewMatrix(
                    cameraEyePosition=[0, 0, 25],
                    cameraTargetPosition=[0, 0, 0],
                    cameraUpVector=[0, 1, 0])
        self.projectionMatrix = p.computeProjectionMatrixFOV(
                    fov=50.0,
                    aspect=1.0,
                    nearVal=0.1,
                    farVal=100.5)

        # self.obstacles = [[self.colBoxId, [5,5,0.1]]]
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        
        # self.reset()

    def get_camera_image(self, sz):
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                                        width=sz, 
                                        height=sz,
                                        viewMatrix=self.viewMatrix,
                                        projectionMatrix=self.projectionMatrix)
        # plt.imshow(segImg); plt.show()
        return width, height, rgbImg, depthImg, segImg

    def get_observation_image(self, sz, agent = "alice"):
        if agent == "alice":
            w,h,_,_,seg = self.get_camera_image(sz)
            obs = (seg>1).astype(float)
            car = (seg==1).astype(float)
            gridmap = np.dstack((car,obs))
        else:
            w,h,_,_,seg = self.get_camera_image(sz)
            obs = (seg>2).astype(float)
            car = (seg==1).astype(float)
            goal = (seg==2).astype(float)
            gridmap = np.dstack((car,obs,goal))
        return gridmap

    def draw_obstacles(self):
        for i in range(len(self.pos_obstacles)):
            boxHalfLength = self.obstacle_dims[i][0]
            boxHalfWidth = self.obstacle_dims[i][1]
            boxHalfHeight = self.obstacle_dims[i][2]
            colBoxId = p.createCollisionShape(p.GEOM_BOX,
                                  halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight])
            p.createMultiBody(baseMass=self.obstacle_mass,
                        baseCollisionShapeIndex=colBoxId,
                        basePosition=self.pos_obstacles[i,:])


    def step(self, action, max_step_multiplier = 1, agent = "alice"):
        # Feed action to the car and get observation of car's state
        # tic = time.time()
        self.car.apply_action(action)
        p.stepSimulation()
        car_ob = self.car.get_observation()

        # Compute reward as L2 change in distance to goal
        # dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
        #                           (car_ob[1] - self.goal[1]) ** 2))
        dist_to_goal = np.linalg.norm(car_ob[:2] - self.goal)
        # reward = max(self.prev_dist_to_goal - dist_to_goal, 0)
        self.prev_dist_to_goal = dist_to_goal

        threshold = 1
        reward = 0
        # Done by running off boundaries
        self.steps += 1
        # if (car_ob[0] >= 10 or car_ob[0] <= -10 or
        #         car_ob[1] >= 10 or car_ob[1] <= -10):
        #     self.done = True
        #     reward = -3
        # Done by reaching goal
        if dist_to_goal < threshold:
            self.done = True
            reward = 5
        if self.steps > max_step_multiplier * self.max_steps:
            reward = -2
            self.done = True
        ob = np.array(car_ob, dtype=np.float32)
        gridmap = self.get_observation_image(75, agent)
        # print("Step time: ", time.time() - tic)
        return gridmap, ob, reward, self.done, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, goal, base_position, base_orientation, agent="alice"):
        #print(agent)
        # tic = time.time()
        self.steps = 0
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)

        # Reload the plane and car
        Plane(self.client)
        self.car = Car(self.client, base_position, base_orientation)

        # Set the goal to a random target
        # x = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else
        #      self.np_random.uniform(-5, -9))
        # y = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else
        #      self.np_random.uniform(-5, -9))
        self.goal = deepcopy(goal)
        self.done = False

        # Visual element of the goal
        Goal(self.client, self.goal)

        # Get observation to return
        car_ob = self.car.get_observation()

        self.prev_dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                           (car_ob[1] - self.goal[1]) ** 2))
        
        # boxHalfLength = 5
        # boxHalfWidth = 5
        # boxHalfHeight = 5
        # colBoxId = p.createCollisionShape(p.GEOM_BOX,
        #                           halfExtents=[boxHalfLength, boxHalfWidth, boxHalfHeight])

        # mass = 100
        # p.createMultiBody(baseMass=mass,
        #             baseCollisionShapeIndex=colBoxId,
        #             basePosition=[5, 5, 0.1])
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        if agent == "alice":
            #If agent is alice, reset obstacle positions
            self.reset_obstacle_positions()

        self.draw_obstacles()
        gridmap = self.get_observation_image(75, agent)
        # print("Reset time: ", time.time() - tic)
        return gridmap, np.array(car_ob, dtype=np.float32)

    def reset_obstacle_positions(self):
        self.pos_obstacles[0][0:2] = np.random.uniform(-5,5,2)

        mult = -1
        for i in range(2):
            mult = -1*mult
            r = np.random.randint(0,3)
            if r == 0:
                self.pos_obstacles[i][0:2] = mult*(np.random.uniform(0,5,2)+np.array([0,5]))
            elif r == 1:
                self.pos_obstacles[i][0:2] = mult*(np.random.uniform(0,5,2)+np.array([5,5]))
            elif r == 2:
                self.pos_obstacles[i][0:2] = mult*(np.random.uniform(0,5,2)+np.array([5,0]))
        mult = -1
        for i in range(2):
            mult = -1*mult
            r = np.random.randint(0,3)
            if r == 0:
                self.pos_obstacles[i+2][0:2] = mult*(np.random.uniform(0,5,2)+np.array([-5,5]))
            elif r == 1:
                self.pos_obstacles[i+2][0:2] = mult*(np.random.uniform(0,5,2)+np.array([-10,5]))
            elif r == 2:
                self.pos_obstacles[i+2][0:2] = mult*(np.random.uniform(0,5,2)+np.array([-10,0]))
    
    def render(self, mode='human'):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        car_id, client_id = self.car.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(car_id, client_id)]
        pos[2] = 0.2

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)

    def close(self):
        p.disconnect(self.client)
