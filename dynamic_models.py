import math
import numpy as np

from lib.utils import clip_to_pi


class DynamicBicycle:
    """
    Canonical Dynamic Bicycle Model
    """
    g = 9.81

    def __init__(self, lf, lr, Caf, Car, Iz, m):
        # model constants
        self.lf = lf  # distance from front axle to tracking point
        self.lr = lr  # distance from rear axle to tracking point
        self.Caf = Caf  # cornering stiffness of each front tire
        self.Car = Car  # cornering stiffness of each rear tire
        self.Iz = Iz  # yaw inertia
        self.m = m  # vehicle mass

        # dynamic parameters
        self.x = 0. # position
        self.y = 0.  # position
        self.yaw = 0.  # heading angle
        self.vx = 0.  # longitudinal speed
        self.vy = 0.  # lateral speed
        self.steering_angle  # wheel angle of front axle

        # state space
        # X = A*X + B*U
        # Y = C*X + D*U
        # X = [y, yd, yaw, yawd] where y = radius of curvature
        # U = [front wheel steering angle, acceleration]
        self.A = np.empty((4, 4))  # depends on longitudinal velocity
        self.B = np.array([
            [0, 0],
            [2*Caf/m, 0],
            [0, 0],
            [2*lf*Caf/Iz, 0],
        ])
        self.C = np.eye(4)  # complete observability
        self.D = np.zeros((4, 2))  # set feedthrough matrix to 0 as usual

    def set_state(self, x, y, yaw):
        """
        set global position and orientation
        """
        self.x = x
        self.y = y
        self.yaw = yaw

    def update_state(self, acceleration, steering_angle, dt):
        """
        update state given parameters:
        - acceleration
        - front wheel steering angle
        - timestep length
        assumes constant velocity
        """
        lf = self.lf
        lr = self.lr
        Caf = self.Caf
        Car = self.Car
        Iz = self.Iz
        m = self.m
        vx = self.vx
        vy = self.vy

        yaw_rate = ??

        # state space representation
        self.A = np.array([
            [0, 1, 0, 0],
            [0, -(2*Caf+2*Car)/(m*vx), 0, -vx-(2*Caf*lf-2*Car*lf)/(m*vx)],
            [0, 0, 0, 1],
            [0, -(2*lf*Caf-2*lr*Car)/(Iz*vx), 0, -(2*lf**2*Caf+2*lr**2+Car)/(Iz*vx)],
        ])

        ax = acceleration + yaw_rate + vy
        self.vx += ax * dt
