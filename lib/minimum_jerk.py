

def calculate_coefficients(start, end, t):
    """
    https://courses.shadmehrlab.org/Shortcourse/minimumjerk.pdf
    Solve for the 6 coefficients that are needed to generate the trajectory.
    Some function that has a 6th derivative of 0 (x(t)'''''' = 0) minimizes jerk.
    The differential equation x(t)'''''' = 0 has a general solution of:

    x(t) = a0 + a1 + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    x(t)' = a1 + 2*a2*t + 3*a3*t^ + 4*a4*t^3 + 5*a5*t^4
    x(t)'' = 2*a2 + 6*a3*t + 12*a4*t^2 + 20*a5*t^3

    Knowns: x(0), x(0)', x(0)'', x(t), x(t)', x(t)''
    6 unknowns, 6 equations -> solve system!

    Parameters
    ----------
    start: [position, velocity, acceleration]
        start of trajectory

    end: [position, velocity, acceleration]
        end of trajectory

    t: float
        total time to traverse the trajectory

    Returns: [a0, a1, a2, a3, a4, a5]
    """
    conditions = np.concatenate((start, end))
    solution_array = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, t, t**2, t**3, t**4, t**5],
        [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4],
        [0, 0, 2, 6*t, 12*t**2, 20*t**3],
    ])
    return np.linalg.inv(solution_array) @ conditions


def calculate_velocity(coefficients, t):
    '''
    Returns velocity at time t in trajectory

    Parameters:
    ----------

    coefficients: np.array
        [a0, a1, a2, a3, a4, a5]

    t: float
        time in trajectory
    '''
    v_array = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4])
    return v_array @ coefficients
