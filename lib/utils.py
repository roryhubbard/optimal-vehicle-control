import math


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
