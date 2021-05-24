import numpy as np
from scipy import signal
from scipy.linalg import solve_discrete_are, inv


def lqr_gains(A, B, Q, R):
    X = solve_discrete_are(A, B, Q, R)
    K = inv(B.T @ X @ B + R) @ B.T @ X @ A
    eigvals = np.linalg.eigvals(A - B @ K)
    return K, X, eigvals


if __name__ == '__main__':
    A = np.eye(4)
    B = np.array([1, 1, 1, 1])[:, np.newaxis]
    C = np.eye(4)
    D = np.zeros((4,1))

    loop_time = .05
    Gcont = signal.StateSpace(A,B,C,D)
    Gdisc = Gcont.to_discrete(loop_time)
    Ad = Gdisc.A
    Bd = Gdisc.B
    Cd = Gdisc.C
    
    Q = np.diag([.001, 1, 1, 1])
    R = np.array([50])

    K, _, _ = lqr_gains(Ad, Bd, Q, R)
