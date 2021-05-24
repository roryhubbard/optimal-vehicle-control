import sys
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

root = str(Path(__file__).parents[1].resolve())
if root not in sys.path:
    sys.path.append(root)

from models.dynamic_models import ErrorBasedDynamicBicycle
from controllers.lqr import lqr_gains


plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['savefig.facecolor'] = 'black'
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['figure.edgecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.titlecolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.rcParams["figure.autolayout"] = True
# plt.rcParams['legend.facecolor'] = 'white'


def closest_node(X, Y, traj):
    point = np.array([X, Y])
    traj = np.asarray(traj)
    dist = point - traj
    dist_2 = np.sum(dist**2, axis=1)
    closest_idx = np.argmin(dist_2)
    return np.sqrt(dist_2[closest_idx]), closest_idx


def main():
    """
    Comparison between LQR and MPC trajectory tracking control
    """
    lr = 1.5
    lf = 1.5
    Ca = 15000.0
    Iz = 3344.0
    m = 2000.0

    dynamic_bike = ErrorBasedDynamicBicycle(lf, lr, Ca, Ca, Iz, m)

    start_x = start_y = start_yaw = 0.
    start_v = 10.  # meters per second

    dynamic_bike.set(X=start_x, Y=start_y, yaw=start_yaw, vx=start_v)

    dynamic_bike_states = [dynamic_bike.observe()]

    v = 10.  # meters per second
    dt = 0.1
    T = 4
    timesteps = [0.]

    trajectory = [(i, 1) for i in range(100)]  # horizontal trajectory
    Q = np.diag([1, 1, 1, 1])
    R = np.diag([1, 1])

    for t in range(1, int(T / dt)):
        A, B, _, _ = dynamic_bike.discretize_state_space(dt)
        K, _, _ = lqr_gains(A, B, Q, R)
        _, closest_idx = closest_node(dynamic_bike.X, dynamic_bike.Y, trajectory)
        X = dynamic_bike.observe_states(trajectory, closest_idx)

        u = -K @ X.T
        steering_angle = u[0]
        accel = u[1]
        dynamic_bike.update(accel, steering_angle, dt)

        dynamic_bike_states.append(dynamic_bike.observe())

        timesteps.append(t*dt)

    fig, ax = plt.subplots(ncols=2)

    dynamic_bike_x = list(map(lambda x: x[0], dynamic_bike_states))
    dynamic_bike_y = list(map(lambda x: x[1], dynamic_bike_states))
    dynamic_bike_yaw = list(map(lambda x: x[2] * 180. / math.pi,
                                dynamic_bike_states))

    ax[0].plot(dynamic_bike_x, dynamic_bike_y, label='LQR')
    ax[0].plot(*list(zip(*trajectory)))
    ax[1].plot(timesteps, dynamic_bike_yaw, label='LQR')

    ax[0].set_title('Position')
    ax[0].set_xlabel('x (m)')
    ax[0].set_ylabel('y (m)')
    # ax[0].set_aspect('equal')
    ax[0].legend(prop={'size': 6})

    ax[1].set_title('Yaw')
    ax[1].set_xlabel('timesteps (s)')
    ax[1].set_ylabel('yaw angle (degrees)')
    ax[1].legend(prop={'size': 6})

    # fig.suptitle('All models used: constance velocity = 10 mph, constant steering angle = 20 degrees')

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
