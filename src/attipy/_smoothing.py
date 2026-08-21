import numpy as np

from ._statespace import _state_transition_update
from ._quatops import _correct_quat_with_rotvec, _correct_quat_with_gibbs2
from ._transforms import _matrix_from_quat, _quat_from_matrix


class RTSSmoother:
    """
    Fixed-interval smoothing for MEKF based on the Rauch-Tung-Striebel (RTS) algorithm.
    """

    def __init__(self, mekf):
        self._mekf = mekf

        # Forward sweep buffers
        self._q_buf = []
        self._b_buf = []
        self._P_buf = []
        self._dtheta_buf = []

        # Smoothed state and covariance estimates
        self._q_nb = np.empty((0, 4), dtype="float64")
        self._bg_b = np.empty((0, 3), dtype="float64")
        self._P = np.empty((0, 6, 6), dtype="float64")

    def update(self, *args, **kwargs):
        """
        Update with IMU and aiding measurements.
        """
        self._mekf.update(*args, **kwargs)
        self._q_buf.append(self._mekf.attitude.as_quaternion())
        self._b_buf.append(self._mekf.bias)
        self._P_buf.append(self._mekf.P)
        self._dtheta_buf.append(self._mekf._dtheta.copy())
        return self

    def _smooth(self):
        n_samples = len(self._q_buf)

        if n_samples == 0:
            pass
        elif n_samples == 1:
            self._q_nb = np.array(self._q_buf)
            self._bg_b = np.array(self._b_buf)
            self._P = np.array(self._P_buf)
        elif n_samples != len(self._q_nb):
            q_nb, bg_b, P = _rts_backward_sweep(
                self._q_buf,
                self._b_buf,
                self._P_buf,
                self._dtheta_buf,
                self._mekf._dx,
                self._mekf._phi,
                self._mekf._Q,
                self._mekf._dt,
            )
            self._q_nb = np.array(q_nb, dtype="float64")
            self._bg_b = np.array(bg_b, dtype="float64")
            self._P = np.array(P, dtype="float64")


def _rts_backward_sweep(q_nb, bg_b, P, dtheta, dx, phi_k, Q, dt):
    """
    Perform a backward sweep with the Rauch-Tung-Striebel (RTS) algorithm.
    """

    q_nb = q_nb.copy()
    bg_b = bg_b.copy()
    P = P.copy()

    q_last_prior = q_nb[-2].copy()
    _correct_quat_with_rotvec(q_last_prior, dtheta[-1])
    q_last_post = q_nb[-1].copy()

    R_last_post = _matrix_from_quat(q_last_post)
    R_last_prior = _matrix_from_quat(q_last_prior)
    dR_last = R_last_post @ R_last_prior.T
    dq_last = _quat_from_matrix(dR_last)
    da_last = 2 * dq_last[1:4] / dq_last[0]

    dx = dx.copy()
    dx[0:3] = da_last
    dx[3:6] = bg_b[-1] - bg_b[-2]

    # Backward sweep
    n = len(q_nb)
    for k in range(n - 2, -1, -1):

        # Update step k state space and calculate a priori covariance for step k + 1
        _state_transition_update(phi_k, dtheta[k])
        P_prior_kp1 = phi_k @ P[k] @ phi_k.T + Q

        # Smoothed error-state estimate and corresponding covariance
        A = P[k] @ phi_k.T @ np.linalg.inv(P_prior_kp1)
        dx = A @ dx
        P[k] += A @ (P[k + 1] - P_prior_kp1) @ A.T

        _correct_quat_with_gibbs2(q_nb[k], dx[0:3])
        bg_b[k] += dx[3:6]

    return q_nb, bg_b, P


    # n_samples = len(q_buf)
    # q_nb = [None] * n_samples
    # bg_b = [None] * n_samples
    # P = [None] * n_samples

    # # Initialize with the last state
    # q_nb[-1] = q_buf[-1]
    # bg_b[-1] = b_buf[-1]
    # P[-1] = P_buf[-1]

    # for k in range(n_samples - 2, -1, -1):
    #     # Compute the smoother gain
    #     P_k = P_buf[k]
    #     P_kp1 = P[k + 1]
    #     phi_k = phi[k]
    #     K_smooth = P_k @ phi_k.T @ np.linalg.inv(P_kp1)

    #     # Update the smoothed state and covariance
    #     dx_smooth = K_smooth @ (dx[k + 1] - phi_k @ dx[k])
    #     q_nb[k] = _quat_from_rotvec(dx_smooth[:3]) @ q_buf[k]
    #     bg_b[k] = b_buf[k] + dx_smooth[3:]
    #     P[k] = P_k + K_smooth @ (P_kp1 - P_k) @ K_smooth.T

    # return q_nb, bg_b, P