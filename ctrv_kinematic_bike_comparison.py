import math
import matplotlib.pyplot as plt
from kinematic_models import CTRVBicycle, FrontWheelSteeringBicycle


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
    Comparison between CTRV and normal Kinematic Bicycle Model
    """
    front_steering_bike = FrontWheelSteeringBicycle(3., 0.)
    ctrv_bike = CTRVBicycle(3., 0.)

    front_steering_x = [front_steering_bike.X]
    front_steering_y = [front_steering_bike.Y]
    front_steering_yaw = [front_steering_bike.yaw]

    ctrv_x = [ctrv_bike.X]
    ctrv_y = [ctrv_bike.Y]
    ctrv_yaw = [ctrv_bike.yaw]

    v = 20 * MPH_to_MS
    steering_angle = 10 * math.pi / 180
    dt = 0.1
    T = 10

    for _ in range(int(T / dt)):
        ctrv_bike.update_state(v, steering_angle, dt)
        ctrv_x.append(ctrv_bike.X)
        ctrv_y.append(ctrv_bike.Y)
        ctrv_yaw.append(ctrv_bike.yaw * 180 / math.pi)

        front_steering_bike.update_state(v, steering_angle, dt)
        front_steering_x.append(front_steering_bike.X)
        front_steering_y.append(front_steering_bike.Y)
        front_steering_yaw.append(front_steering_bike.yaw * 180 / math.pi)

    fig, ax = plt.subplots(ncols=2)

    ax[0].plot(front_steering_x, front_steering_y, label='Kinematic Bicycle')
    ax[1].plot(front_steering_yaw, label='Kinematic Bicycle')
    ax[0].plot(ctrv_x, ctrv_y, label='CTRV')
    ax[1].plot(ctrv_yaw,'--', label='CTRV')

    ax[0].set_title('Position')
    ax[0].set_xlabel('x (m)')
    ax[0].set_ylabel('y (m)')
    ax[0].set_aspect('equal')
    ax[0].legend()

    ax[1].set_title('Yaw')
    ax[1].set_xlabel('timesteps')
    ax[1].set_ylabel('yaw angle (degrees)')
    ax[1].legend()

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
