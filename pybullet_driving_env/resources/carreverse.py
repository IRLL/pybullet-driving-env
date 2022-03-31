import pybullet as p
import numpy as np
from pybullet_driving_env.resources.car import Car

class CarReverse(Car):
    def __init__(self, client, base_position, base_orientation):
        super().__init__(client, base_position, base_orientation)

    def apply_action(self, action):
        # Expects action to be two dimensional
        throttle, steering_angle, reverse = action

        # Clip throttle and steering angle to reasonable values
        throttle = min(max(throttle, 0), 1)
        reverse = min(max(reverse, 0), 1)
        steering_angle = max(min(steering_angle, 0.6), -0.6)
        if reverse > 0.5:
            throttle = -0.2*throttle

        # Set the steering joint positions
        p.setJointMotorControlArray(self.car, self.steering_joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=[steering_angle] * 2,
                                    physicsClientId=self.client)

        # Calculate drag / mechanical resistance ourselves
        # Using velocity control, as torque control requires precise models
        friction = -self.joint_speed * (self.joint_speed * self.c_drag +
                                        self.c_rolling)
        acceleration = self.c_throttle * throttle + friction
        # Each time step is 1/240 of a second
        self.joint_speed = self.joint_speed + 1/30 * acceleration
        # if self.joint_speed < 0:
        #     self.joint_speed = 0

        # Set the velocity of the wheel joints directly
        p.setJointMotorControlArray(
            bodyUniqueId=self.car,
            jointIndices=self.drive_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[self.joint_speed] * 4,
            forces=[1.2] * 4,
            physicsClientId=self.client)