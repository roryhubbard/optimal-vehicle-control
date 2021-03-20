import math

from lib.utils import clip_to_pi


class KinematicBicycle:
    """
    Canonical Kinematic Bicycle Model
    """

    def __init__(self, lf, lr):
        self.lf = lf  # distance from front axle to tracking point
        self.lr = lr  # distance from rear axle to tracking point
        self.x = 0. # position
        self.y = 0.  # position
        self.yaw = 0.  # heading angle
        self.v = 0.  # speed
        self.delta_f = 0.  # wheel angle of front axle
        self.delta_r = 0.  # wheel angle of front axle
        self.beta = 0.  # slip angle

    def set_state(self, x, y, yaw):
        """
        set global position and orientation
        """
        self.x = x
        self.y = y
        self.yaw = yaw

    def update_state(self, v, delta_f, delta_r, dt):
        """
        update state given parameters:
        - velocity
        - front wheel steering angle
        - rear wheel steering angle
        - timestep length
        assumes constant velocity
        """
        yaw_rate = v * math.cos(self.beta) / (self.lf + self.lr) \
            * (math.tan(delta_f) - math.tan(delta_r))

        self.x += v * math.cos(self.yaw + self.beta) * dt
        self.y += v * math.sin(self.yaw + self.beta) * dt

        self.yaw += yaw_rate * dt
        self.yaw = clip_to_pi(self.yaw)  # [-pi, pi]

        self.v = v
        self.delta_f = delta_f
        self.delta_r = delta_r
        self.beta = math.atan2(
            self.lf * math.tan(delta_r) + self.lr * math.tan(delta_f),
            self.lf + self.lr)


class FrontWheelSteeringBicycle(KinematicBicycle):
    """
    Assumptions:
    - 0 slip angle
    - 0 rear tire steering angle
    """

    def update_state(self, v, steering_angle, dt):
        """
        same update calculation and KinematicBicycle but rear wheel steering
        angle will always be set to 0
        """
        super().update_state(v, steering_angle, 0., dt)


class CTRVBicycle(FrontWheelSteeringBicycle):
    """
    Constant Turn Rate and Velocity
    Assumptions:
    - same as FrontWheelSteeringBicycle
    """

    def update_state(self, v, steering_angle, dt):
        """
        CTRV update calculation incorporates the yaw rate in the position
        update of the KinematicBicycle, making it more accurate as the turn
        becomes sharper
        """
        yaw_rate = v / (self.lf + self.lr) * math.tan(steering_angle)

        # only consider yaw rate in position update if it is nonzero
        if abs(yaw_rate) > 0.01:
            self.x += v / yaw_rate * (math.sin(self.yaw + yaw_rate * dt)
                                      - math.sin(self.yaw))
            self.y += v / yaw_rate * (math.cos(self.yaw)
                                      - math.cos(self.yaw + yaw_rate * dt))

            self.yaw += yaw_rate * dt
            self.yaw = clip_to_pi(self.yaw)  # [-pi, pi]

            self.v = v
            self.delta_f = steering_angle
        else:
            # if yaw rate is sufficiently small then just use normal kinematic
            # bicycle model for updating state
            super().update_state(v, steering_angle, 0., dt)
