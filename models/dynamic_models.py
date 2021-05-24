import sys
from pathlib import Path
from math import sin, cos, atan2
import numpy as np
from scipy import signal

root = str(Path(__file__).parents[1].resolve())
if root not in sys.path:
    sys.path.append(root)

from lib.utils import clip_to_pi
from models.model_templates import VehicleModel


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

        # lateral dynamics state space
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
        self.C = np.eye(4)
        self.D = np.zeros((4, 2))

    def update(self, accel, delta, dt):
        """
        Update vehicle state

        Input:
        - accel: longitudinal acceleration
        - delta: front wheel steering angle
        - dt: timestep length
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
        # y is technically radius of curvature but it doesn't contribute to
        # the update so just set it to 0
        X = np.array([0., vy, yaw, yawd])
        if vx > 0.5:
            self.A = np.array([
                [0, 1, 0, 0],
                [0, -2*(Caf+Car)/(m*vx), 0, -vx+(2*Car*lr-2*Caf*lf)/(m*vx)],
                [0, 0, 0, 1],
                [0, (2*lr*Car-2*lf*Caf)/(Iz*vx), 0, -(2*lf**2*Caf+2*lr**2*Car)/(Iz*vx)],
            ])
        else:
            # at low speeds the model is unstable
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


class ErrorBasedDynamicBicycle(DynamicBicycle):
    """
    Error Based Dynamic Bicycle Model

    Assumptions:
    - same as DynamicBicycle
    - desired yaw rate is 0
    """

    def discretize_state_space(self, dt):
        """
        Calculate discrete state space matrices

        Input:
        - dt: timestep length

        Output:
        - discrete state space matrices A, B, C, D
        """
        # model constants
        lf = self.lf
        lr = self.lr
        Caf = self.Caf
        Car = self.Car
        Iz = self.Iz
        m = self.m

        vx = self.vx

        self.A = np.array([
            [0, 1, 0, 0],
            [0, -2*(Caf+Car)/(m*vx), 2*(Caf+Car)/m, 2*(Car*lr-Caf*lf)/(m*vx)],
            [0, 0, 0, 1],
            [0, -2*(lf*Caf-lr*Car)/(Iz*vx), 2*(Caf*lf-Car*lr)/Iz,
             -2*(lf**2*Caf+lr**2*Car)/(Iz*vx)],
        ])

        Gcont = signal.StateSpace(self.A, self.B, self.C, self.D)
        Gdisc = Gcont.to_discrete(dt)
        return Gdisc.A, Gdisc.B, Gdisc.C, Gdisc.D

    def observe_states(self, traj, closest_idx):
        """
        Calculate error based states
        """
        if closest_idx<len(traj)-10:
            idx_fwd = 10
        else:
            idx_fwd = len(traj)-closest_idx-1
        yawdes = atan2((traj[closest_idx+idx_fwd][1]-self.Y),
                       (traj[closest_idx+idx_fwd][0]-self.X))
        # yawdes = self.vx*curv[closest_idx+idx_fwd]
        yawdesdot = 0
        X = np.zeros(4)
        X[0] = (self.Y - traj[closest_idx+idx_fwd][1]) * cos(yawdes) \
               - (self.X - traj[closest_idx+idx_fwd][0]) * sin(yawdes)
        X[2] = clip_to_pi(self.yaw - yawdes)
        X[1] = self.vy + self.vx*X[2]
        X[3] = self.yawd - yawdesdot

        return X

# def compute_curvature(self):
# """
# Function to compute and return the curvature of trajectory.
# """
# sigma_gaus = 10
# 5traj=self.traj
# xp = scipy.ndimage.filters.gaussian_filter1d(input=traj[:,0],
# sigma=sigma_gaus,order=1)
# xpp = scipy.ndimage.filters.gaussian_filter1d(input=traj[:,0],
# sigma=sigma_gaus,order=2)
# yp = scipy.ndimage.filters.gaussian_filter1d(input=traj[:,1],
# sigma=sigma_gaus,order=1)
# ypp = scipy.ndimage.filters.gaussian_filter1d(input=traj[:,1],
# sigma=sigma_gaus,order=2)
# curv=np.zeros(len(traj))
# for i in range(len(xp)):
# curv[i] = (xp[i]*ypp[i] - yp[i]*xpp[i])/(xp[i]**2 + yp[i]**2)**1.5
# return curv
