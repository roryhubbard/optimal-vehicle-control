import math
import sys
from pathlib import Path
import numpy as np

root = str(Path(__file__).parents[1].resolve())
if root not in sys.path:
    sys.path.append(root)

from lib.utils import rotate_and_translate
from controllers.simple_pid import SimplePID


class SpeedController:

    def __init__(self, Kp, Ki, Kd):
        self.pid_controller = SimplePID(Kp, Ki, Kd)

    def point_follow_control(self, target_speed, target_pos,
                             veh_speed, veh_pos, veh_yaw, dt):
        dx_global = target_pos[0] - veh_pos[0]
        dy_global = target_pos[1] - veh_pos[1]
        dx_local, dy_local = rotate_and_translate(
            np.array([dx_global, dy_global]), -veh_yaw)
        speed_error = target_speed - veh_speed
        accel = speed_error + self.pid_controller.get_control_input(dx_local, dt)
        return max(0, veh_speed + accel * dt)
