import math
import matplotlib.pyplot as plt
from dynamic_models import DynamicBicycle
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
plt.rcParams['text.color'] = 'white'
plt.rcParams["figure.autolayout"] = True
# plt.rcParams['legend.facecolor'] = 'white'


MPH_to_MS = 0.447


def main():
    """
    Comparison between Dynamic Bicycle, CTRV Bicycle, and Front Steering
    Kinematic Bicycle
    """
    lr = 1.5
    lf = 1.5
    Ca = 15000.0
    Iz = 3344.0
    m = 2000.0

    dynamic_bike = DynamicBicycle(lf, lr, Ca, Ca, Iz, m)
    front_steering_bike = FrontWheelSteeringBicycle(3., 0.)
    ctrv_bike = CTRVBicycle(3., 0.)

    start_x = start_y = start_yaw = 0.
    start_v = 20 * MPH_to_MS

    dynamic_bike.set(X=start_x, Y=start_y, yaw=start_yaw, vx=start_v)
    front_steering_bike.set(X=start_x, Y=start_y, yaw=start_yaw, v=start_v)
    ctrv_bike.set(X=start_x, Y=start_y, yaw=start_yaw, v=start_v)

    dynamic_bike_states = [dynamic_bike.observe()]
    front_steering_states = [front_steering_bike.observe()]
    ctrv_bike_states = [ctrv_bike.observe()]

    v = 20. * MPH_to_MS
    accel = 0.
    steering_angle = 20. * math.pi / 180.
    dt = 0.1
    T = 5
    timesteps = [0.]

    for t in range(1, int(T / dt)):
        dynamic_bike.update(accel, steering_angle, dt)
        front_steering_bike.update(v, steering_angle, dt)
        ctrv_bike.update(v, steering_angle, dt)

        dynamic_bike_states.append(dynamic_bike.observe())
        front_steering_states.append(front_steering_bike.observe())
        ctrv_bike_states.append(ctrv_bike.observe())

        timesteps.append(t*dt)

    fig, ax = plt.subplots(ncols=2)

    dynamic_bike_x = list(map(lambda x: x[0], dynamic_bike_states))
    dynamic_bike_y = list(map(lambda x: x[1], dynamic_bike_states))
    dynamic_bike_yaw = list(map(lambda x: x[2] * 180. / math.pi,
                                dynamic_bike_states))

    front_steering_x = list(map(lambda x: x[0], front_steering_states))
    front_steering_y = list(map(lambda x: x[1], front_steering_states))
    front_steering_yaw = list(map(lambda x: x[2] * 180. / math.pi,
                                front_steering_states))

    ctrv_x = list(map(lambda x: x[0], ctrv_bike_states))
    ctrv_y = list(map(lambda x: x[1], ctrv_bike_states))
    ctrv_yaw = list(map(lambda x: x[2] * 180. / math.pi,
                                ctrv_bike_states))

    ax[0].plot(dynamic_bike_x, dynamic_bike_y, label='Dynamic Bicycle')
    ax[1].plot(timesteps, dynamic_bike_yaw, label='Dynamic Bicycle')

    ax[0].plot(front_steering_x, front_steering_y,
               label='Front Steering Kinematic Bicycle')
    ax[1].plot(timesteps, front_steering_yaw,
               label='Front Steering Kinematic Bicycle')

    ax[0].plot(ctrv_x, ctrv_y, label='CTRV Kinematic Bicycle')
    ax[1].plot(timesteps, ctrv_yaw,'--', label='CTRV Kinematic Bicycle')

    ax[0].set_title('Position')
    ax[0].set_xlabel('x (m)')
    ax[0].set_ylabel('y (m)')
    ax[0].set_aspect('equal')
    ax[0].legend(prop={'size': 6})

    ax[1].set_title('Yaw')
    ax[1].set_xlabel('timesteps (s)')
    ax[1].set_ylabel('yaw angle (degrees)')
    ax[1].legend(prop={'size': 6})

    fig.suptitle('All models used: constance velocity = 20 mph, constant steering angle = 20 degrees')

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
