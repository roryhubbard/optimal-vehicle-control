import math
import numpy as np


def get_wheel_angle(curvature, wheel_base):
    '''
    return front wheel steering angle given vehicle curvature and wheel base
    '''
    return math.atan(curvature * wheel_base)


def get_curvature(wheel_angle, wheel_base):
    '''
    return vehicle curvature given front wheel steering angle and wheel base
    '''
    return math.tan(wheel_angle) / wheel_base


def clip_to_pi(theta):
    '''
    restrict theta to [-pi, pi]
    '''
    return math.atan2(math.sin(theta), math.cos(theta))


def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def rotate_and_translate(x: np.ndarray, theta: float, translation=np.zeros(2)):
    '''
    x: (N, 2)
    '''
    return x @ np.array([
        [math.cos(theta), math.sin(theta)],
        [-math.sin(theta), math.cos(theta)],
    ]) + translation
