import time


class SimplePID:

    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.error_integral = 0
        self.last_error = None

    def get_control_input(self, error, dt):
        self.error_integral += error
        error_derivative = (error - self.last_error) / dt \
            if self.last_error is not None \
            else 0.
        self.last_error = error
        return self.Kp * error + self.Ki * self.error_integral \
            + self.Kd * error_derivative
