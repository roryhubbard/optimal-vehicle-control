import math
import matplotlib.pyplot as plt
from dynamic_models import DynamicBicycle


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
# plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.rcParams["figure.autolayout"] = True


MPH_to_MS = 0.447


def main():
    """
    Validation of Dynamic Bicycle Model
    """
    lr = 1.5
    lf = 1.5
    Ca = 15000.0
    Iz = 3344.0
    m = 2000.0
    dynamic_bike = DynamicBicycle(lf, lr, Ca, Ca, Iz, m)
    dynamic_bike.vx = 20 * MPH_to_MS

    dynamic_bike_x = [dynamic_bike.X]
    dynamic_bike_y = [dynamic_bike.Y]
    dynamic_bike_yaw = [dynamic_bike.yaw]

    accel = 0.
    steering_angle = 10 * math.pi / 180
    dt = 0.1
    T = 10

    for _ in range(int(T / dt)):
        dynamic_bike.update_state(steering_angle, accel, dt)
        dynamic_bike_x.append(dynamic_bike.X)
        dynamic_bike_y.append(dynamic_bike.Y)
        dynamic_bike_yaw.append(dynamic_bike.yaw * 180 / math.pi)

    fig, ax = plt.subplots(ncols=2)

    ax[0].plot(dynamic_bike_x, dynamic_bike_y, label='Dynamic Bicycle')
    ax[1].plot(dynamic_bike_yaw, '--', label='Dynamic Bicycle')

    ax[0].set_title('Position')
    ax[0].set_xlabel('x (m)')
    ax[0].set_ylabel('y (m)')
    ax[0].set_aspect('equal')

    ax[1].set_title('Yaw')
    ax[1].set_xlabel('timesteps')
    ax[1].set_ylabel('yaw angle (degrees)')

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
