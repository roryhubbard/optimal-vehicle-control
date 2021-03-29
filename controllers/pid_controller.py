import time


class PIDController:

    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.error_integral = 0
        self.last_error = 0
        self.t = 0

    def get_control_input(self, error):
        curr_t = time.time()
        dt = curr_t - self.t
        self.error_integral += error
        error_derivative = (error - self.last_error) / dt
        self.t = curr_t
        self.last_error = error
        return self.Kp * error + self.Ki * self.error_integral \
            + self.Kd * error_derivative
