"""
Target motion model and Kalman filter.

Matches Musicki & Evans (2002) JIPDA paper, Section 4:
  x(k+1) = F x(k) + v(k)
  x = [x, xdot, y, ydot]^T
  F = blockdiag(F_T, F_T),  F_T = [[1, T], [0, 1]]
  Q = q * blockdiag(Q_T, Q_T),  Q_T = [[T^4/4, T^3/2], [T^3/2, T^2]]
  T = 1s sampling period, q = 0.75 (process noise intensity)
  Measurement: position only (x, y), RMS sensor noise = 5m per axis
"""
import numpy as np


class ConstantVelocityModel:
    """CV motion model + measurement model, matching the JIPDA paper's params."""

    def __init__(self, T=1.0, q=0.75, meas_std=5.0):
        self.T = T
        self.q = q
        self.meas_std = meas_std

        F_T = np.array([[1, T], [0, 1]])
        self.F = np.block([
            [F_T, np.zeros((2, 2))],
            [np.zeros((2, 2)), F_T]
        ])

        Q_T = np.array([[T**4 / 4, T**3 / 2],
                         [T**3 / 2, T**2]])
        self.Q = q * np.block([
            [Q_T, np.zeros((2, 2))],
            [np.zeros((2, 2)), Q_T]
        ])

        # Measurement matrix: observe (x, y) positions only
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])
        self.R = (meas_std ** 2) * np.eye(2)

    def step(self, x, rng):
        """Propagate true state one timestep with process noise."""
        w = rng.multivariate_normal(np.zeros(4), self.Q)
        return self.F @ x + w

    def measure(self, x, rng):
        """Generate a noisy measurement of true state x."""
        v = rng.multivariate_normal(np.zeros(2), self.R)
        return self.H @ x + v


class KalmanFilter:
    """Standard Kalman filter for the CV model, used inside JIPDA track update."""

    def __init__(self, model: ConstantVelocityModel):
        self.model = model

    def predict(self, x, P):
        F, Q = self.model.F, self.model.Q
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        return x_pred, P_pred

    def innovation(self, x_pred, P_pred, z):
        """Return innovation, innovation covariance, and Kalman gain."""
        H, R = self.model.H, self.model.R
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        return y, S, K

    def update(self, x_pred, P_pred, z):
        """Standard single-measurement Kalman update."""
        H = self.model.H
        y, S, K = self.innovation(x_pred, P_pred, z)
        x_new = x_pred + K @ y
        P_new = (np.eye(4) - K @ H) @ P_pred
        return x_new, P_new

    def pda_update(self, x_pred, P_pred, measurements, betas, beta0):
        """
        PDA-style update combining multiple candidate measurements
        weighted by association probabilities beta_i, with beta0 =
        probability none originated from this track.

        measurements: list of z (2,) arrays in this track's gate
        betas: list of beta_i, same length/order as measurements
        beta0: probability of no detection

        Returns combined (x_new, P_new).
        """
        H = self.model.H
        if len(measurements) == 0:
            return x_pred, P_pred

        S = H @ P_pred @ H.T + self.model.R
        K = P_pred @ H.T @ np.linalg.inv(S)

        innovations = [z - H @ x_pred for z in measurements]
        combined_innov = sum(b * y for b, y in zip(betas, innovations))
        x_new = x_pred + K @ combined_innov

        # Spread-of-innovations covariance (standard PDA formula)
        P_c = P_pred - K @ S @ K.T
        sum_b_yy = sum(
            b * np.outer(y, y) for b, y in zip(betas, innovations)
        )
        P_tilde = K @ (sum_b_yy - np.outer(combined_innov, combined_innov)) @ K.T
        P_new = beta0 * P_pred + (1 - beta0) * P_c + P_tilde
        return x_new, P_new
