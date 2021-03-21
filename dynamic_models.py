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
        self.X = 0. # global position
        self.Y = 0.  # global position
        self.yaw = 0.  # yaw angle
        self.yawd = 0.  # yaw rate
        self.vx = 0.  # longitudinal speed
        self.vy = 0.  # lateral speed
        self.delta  # wheel angle of front axle

        # later state space
        # X = A*X + B*U
        # Y = C*X + D*U
        # X = [y, yd, yaw, yawd]
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

    def set_state(self, X, Y, yaw):
        """
        set global position and orientation
        """
        self.X = X
        self.Y = Y
        self.yaw = yaw

    def update_state(self, delta, accel, dt):
        """
        update state given parameters:
        - front wheel steering angle
        - acceleration
        - timestep length
        assumes constant velocity
        """
        # model constants
        lf = self.lf
        lr = self.lr
        Caf = self.Caf
        Car = self.Car
        Iz = self.Iz
        m = self.m

        vx = max(self.vx, 0.01)  # appears in denominator
        vy = self.vy
        yaw = self.yaw
        yawd = self.yawd

        # [y, yd, yaw, yawd]
        # y is technically the radius of curvature but it doesn't contribute to
        # the update so just set it to 0
        X = np.array([0, vy, yaw, yawd])
        self.A = np.array([
            [0, 1, 0, 0],
            [0, -(2*Caf+2*Car)/(m*vx), 0, -vx-(2*Caf*lf-2*Car*lf)/(m*vx)],
            [0, 0, 0, 1],
            [0, -(2*lf*Caf-2*lr*Car)/(Iz*vx), 0, -(2*lf**2*Caf+2*lr**2+Car)/(Iz*vx)],
        ])
        U = np.array([delta, accel])

        Xd = self.A @ X + self.B @ U
        Y = self.C @ X + self.D @ U

        self.yaw += Xd[3] * dt
        self.yawd += Xd[4] * dt

        # TODO
        self.X += 0.
        self.Y = 0.  # global position
        self.vx = 0.  # longitudinal speed
        self.vy = 0.  # lateral speed
        self.delta = 0.  # wheel angle of front axle
