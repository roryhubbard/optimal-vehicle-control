import sys
import math
from pathlib import Path
import numpy as np

root = str(Path(__file__).parents[1].resolve())
if root not in sys.path:
    sys.path.append(root)

from lib.utils import euclidean_distance, rotate_and_translate, get_wheel_angle


class PurePursuitController():
    """
    https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf
    """

    def __init__(self, wheel_base):
        self.wheel_base = wheel_base

    def get_wheel_angle(self, target_pos, veh_pos, veh_yaw):
        dx_global = target_pos[0] - veh_pos[0]
        dy_global = target_pos[1] - veh_pos[1]
        dx_local, dy_local = rotate_and_translate(
            np.array([dx_global, dy_global]), -veh_yaw)
        l = euclidean_distance(target_pos, veh_pos)
        target_curvature = 2 * dy_local / l**2

        return get_wheel_angle(target_curvature, self.wheel_base)
