import sys
import math
from pathlib import Path
import numpy as np

root = str(Path(__file__).parents[1].resolve())
if root not in sys.path:
    sys.path.append(root)

from lib.utils import clip_to_pi, rotate_and_translate

class StanleyController():
    """
    https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf
    """

    def __init__(self, k, delta_max):
        self.k = k  # gain parameter
        self.delta_max = delta_max  # max steering angle

    def steering_control(self, target_pos, target_yaw, veh_pos, veh_yaw, veh_speed):
        yaw_error = clip_to_pi(target_yaw - veh_yaw)
        dx_global = target_pos[0] - veh_pos[0]
        dy_global = target_pos[1] - veh_pos[1]
        _dx_local, dy_local = rotate_and_translate(
            np.array([dx_global, dy_global]), -target_yaw)

        u = yaw_error + math.atan(self.k * dy_local / (.5 + veh_speed))
        return min(max(u, -self.delta_max), self.delta_max)
