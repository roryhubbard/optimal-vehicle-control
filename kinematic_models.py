import math

from lib.utils import clip_to_pi
from model_templates import VehicleModel


class KinematicBicycle(VehicleModel):
    """
    Canonical Kinematic Bicycle Model

    Assumptions:
    - velocity at each wheel is in the direction of the wheel
    """

    def __init__(self, lf, lr):
        # model constants
        self.lf = lf  # distance from front axle to tracking point
        self.lr = lr  # distance from rear axle to tracking point

        # dynamic parameters
        self.beta = 0.  # slip angle
        self.v = 0.  # velocity
        self.delta_f = 0.  # wheel angle of front axle
        self.delta_r = 0.  # wheel angle of front axle

    def update(self, v, delta_f, delta_r, dt):
        """
        Update vehicle state

        Input:
        - v: velocity
        - delta_f: front wheel steering angle
        - delta_r: rear wheel steering angle
        - dt: timestep length

        Assumptions:
        - zero acceleration
        """
        yaw_rate = v * math.cos(self.beta) / (self.lf + self.lr) \
            * (math.tan(delta_f) - math.tan(delta_r))

        self.X += v * math.cos(self.yaw + self.beta) * dt
        self.Y += v * math.sin(self.yaw + self.beta) * dt

        self.yaw += yaw_rate * dt
        self.yaw = clip_to_pi(self.yaw)

        self.v = v
        self.delta_f = delta_f
        self.delta_r = delta_r
        self.beta = math.atan2(
            self.lf * math.tan(delta_r) + self.lr * math.tan(delta_f),
            self.lf + self.lr)


class FrontWheelSteering(KinematicBicycle):
    """
    Only the front axle is used to steer

    Assumptions:
    - same as parent
    - 0 slip angle
    - 0 rear tire steering angle
    """

    def update(self, v, delta, dt):
        """
        Same update calculation as KinematicBicycle but rear wheel steering
        angle will always be set to 0
        """
        super().update(v, delta, 0., dt)


class RearWheelSteering(KinematicBicycle):
    """
    Only the real axle is used to steer

    Assumptions:
    - same as parent
    - 0 slip angle
    - 0 front tire steering angle
    """

    def update(self, v, delta, dt):
        """
        Same update calculation as KinematicBicycle but front wheel steering
        angle will always be set to 0
        """
        super().update(v, 0., delta, dt)


class FourWheelSteering(KinematicBicycle):
    """
    Both axles are used to steer. The front axle is always equal and
    opposite of the rear axle. Allows for smaller turn radiuses for the same
    steering angle.

    Assumptions:
    - same as parent
    - 0 slip angle
    - front tire and rear tire steering angle are equal in magnitude and
      opposite in direction
    """

    def update(self, v, delta, dt):
        """
        Same update calculation as KinematicBicycle but front wheel and rear
        wheel will always be equal and opposite
        """
        super().update(v, delta, -delta, dt)


class CrabSteering(KinematicBicycle):
    """
    Both axles are used to steer. The front axle and rear axle turn in lockstep.
    Allows for translations with zero yaw rotation.

    Assumptions:
    - same as parent
    - 0 slip angle
    - front tire and rear tire steering angle are euqal in magnitude and
      direction
    """

    def update(self, v, delta, dt):
        """
        Same update calculation as KinematicBicycle but front wheel and rear
        wheel will always be exactly equal
        """
        super().update(v, delta, delta, dt)


class CTRV(FrontWheelSteering):
    """
    Constant Turn Rate and Velocity

    Assumptions:
    - same as parent
    - constant yaw rate and velocity during updates
    """

    def update(self, v, delta, dt):
        """
        CTRV update calculation incorporates the yaw rate in the position
        update of the KinematicBicycle, making it more accurate as the turn
        becomes sharper
        """
        yaw_rate = v / (self.lf + self.lr) * math.tan(delta)

        # only consider yaw rate in position update if it is nonzero
        if abs(yaw_rate) > 0.01:
            self.X += v / yaw_rate * (math.sin(self.yaw + yaw_rate * dt)
                                      - math.sin(self.yaw))
            self.Y += v / yaw_rate * (math.cos(self.yaw)
                                      - math.cos(self.yaw + yaw_rate * dt))

            self.yaw += yaw_rate * dt
            self.yaw = clip_to_pi(self.yaw)

            self.v = v
            self.delta_f = delta
        else:
            # if yaw rate is sufficiently small then just use normal kinematic
            # bicycle model for updating state
            super().update(v, delta, 0., dt)
