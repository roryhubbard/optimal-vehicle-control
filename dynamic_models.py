from math import sin, cos
import numpy as np

from lib.utils import clip_to_pi
from model_templates import VehicleModel


class DynamicBicycle(VehicleModel):
    """
    Canonical 2 Degrees of Freedom Dynamic Bicycle Model

    Assumptions:
    - zero aerodynamic friction
    """

    def __init__(self, lf, lr, Caf, Car, Iz, m):
        # model constants
        self.lf = lf  # distance from front axle to center of gravity
        self.lr = lr  # distance from rear axle to center of gravity
        self.Caf = Caf  # cornering stiffness of each front tire
        self.Car = Car  # cornering stiffness of each rear tire
        self.Iz = Iz  # yaw inertia
        self.m = m  # vehicle mass

        # dynamic parameters
        self.yawd = 0.  # yaw rate
        self.vx = 0.  # longitudinal velocity
        self.vy = 0.  # lateral velocity
        self.delta = 0.  # wheel angle of front axle

        # state space
        # X = A*X + B*U
        # Y = C*X + D*U
        # X = [y, yd, yaw, yawd]  # where y = radius of curvature
        # U = [front wheel steering angle, acceleration]
        self.A = np.empty((4, 4))  # depends on longitudinal velocity
        self.B = np.array([
            [0, 0],
            [2*Caf/m, 0],
            [0, 0],
            [2*lf*Caf/Iz, 0],
        ])

    def update(self, accel, delta, dt):
        """
        Update vehicle state

        Input:
        - accel: longitudinal acceleration
        - delta: front wheel steering angle
        - dt: timestep length
        assumes constant velocity
        """
        # model constants
        lf = self.lf
        lr = self.lr
        Caf = self.Caf
        Car = self.Car
        Iz = self.Iz
        m = self.m

        vx = self.vx
        vy = self.vy
        yaw = self.yaw
        yawd = self.yawd

        self.delta = delta

        # [y, yd, yaw, yawd]
        # y is radius of curvature but it doesn't contribute to
        # the update so just set it to 0
        X = np.array([0., vy, yaw, yawd])
        if vx > 0.5:
            self.A = np.array([
                [0, 1, 0, 0],
                [0, -(2*Caf+2*Car)/(m*vx), 0, -vx-(2*Caf*lf-2*Car*lr)/(m*vx)],
                [0, 0, 0, 1],
                [0, -(2*lf*Caf-2*lr*Car)/(Iz*vx), 0, -(2*lf**2*Caf+2*lr**2*Car)/(Iz*vx)],
            ])
        else:
            # at low speeds the dynamic bicycle model is unstable
            self.A = np.array([
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ])

        U = np.array([self.delta, accel])
        Xd = self.A @ X + self.B @ U

        self.yawd += Xd[3] * dt
        self.yaw += self.yawd * dt
        self.yaw = clip_to_pi(self.yaw)

        self.vy += Xd[1] * dt
        self.vx += (self.yawd * self.vy + accel) * dt

        self.X += (self.vx * cos(self.yaw) - self.vy * sin(self.yaw)) * dt
        self.Y += (self.vx * sin(self.yaw) + self.vy * cos(self.yaw)) * dt

    def observe(self):
        """
        Return observable states

        Output:
        - (global x position, global y position, yaw angle)
        """
        return (self.X, self.Y, self.yaw)
