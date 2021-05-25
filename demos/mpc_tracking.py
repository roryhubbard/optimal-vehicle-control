import sys
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

root = str(Path(__file__).parents[1].resolve())
if root not in sys.path:
    sys.path.append(root)

from models.dynamic_models import DynamicBicycle
from controllers.mpc import MPC


class Simulator:

    def __init__(self, A, B, C, Q, R, RD, umin, umax, N):
        self.A = A
        self.B = B
        self.C = C
        self.num_outputs = C.shape[0]
        self.num_inputs = B.shape[1]

        self.mpc = MPC(A, B, C, Q, R, RD, umin, umax, N)

        plt.rcParams['savefig.facecolor'] = 'xkcd:black'

    def get_reference_trajectory(self, n):
        self.t = np.linspace(1, n, n)

        ry = np.zeros(len(self.t))
        ryd = np.zeros(len(self.t))
        # ryaw = signal.square(self.t / 16)
        ryaw = np.ones(len(self.t))
        ryawd = np.zeros(len(self.t))

        self.ref_traj = np.row_stack((ry, ryd, ryaw, ryawd))
    
    def simulate(self):
        U, X = self.establish_starting_state()

        for i in range(self.ref_traj.shape[1]):

            if i == 0:
                self.X_hist = X
                self.U_hist = U
                self.Y_hist = self.C @ X

            else:
                self.X_hist = np.column_stack((self.X_hist, X))
                self.U_hist = np.column_stack((self.U_hist, U))
                self.Y_hist = np.column_stack((self.Y_hist, self.C @ X))
            
            remaining_traj = self.ref_traj[:, i:]

            U = self.mpc.get_control_input(X, U, remaining_traj)

            X = self.update_states(X, U)

    def establish_starting_state(self):
        U = np.zeros((self.B.shape[1], 1))
        X = np.zeros((self.A.shape[1], 1))

        return U, X

    def update_states(self, X, U):
        X = self.A @ X + self.B @ U
        
        return X


def main():
    lr = 1.5
    lf = 1.5
    Ca = 15000.0
    Iz = 3344.0
    m = 2000.0

    dynamic_bike = DynamicBicycle(lf, lr, Ca, Ca, Iz, m)
    start_x = start_y = start_yaw = 0.
    start_v = 5.
    dynamic_bike.set_state_transition_matrix(start_v)
    dynamic_bike.set(X=start_x, Y=start_y, yaw=start_yaw, vx=start_v)

    A = dynamic_bike.A
    B = dynamic_bike.B
    C = dynamic_bike.C

    Q = np.diag(np.array([1., 1., 1., 1.]))
    R = np.diag(np.array([1., 1.]))
    RD = np.diag(np.array([1., 1.]))

    umin = np.array([-math.pi/3, -1])[:, np.newaxis]
    umax = np.array([math.pi/3, 1])[:, np.newaxis]

    N = 10
    
    sim = Simulator(A, B, C, Q, R, RD, umin, umax, N)

    traj_length = 4 * N

    sim.get_reference_trajectory(traj_length)

    sim.simulate()


if __name__ == '__main__':
    main()
