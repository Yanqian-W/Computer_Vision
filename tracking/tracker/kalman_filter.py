import numpy as np
from filterpy.kalman import KalmanFilter


def xywh_to_z(bbox):
    """
    Convert [x, y, w, h] bounding box to Kalman filter state format [cx, cy, s, r].
    - cx, cy: centre of box
    - s: scale = area
    - r: aspect ratio = w / h
    """
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    s = w * h         # scale (area)
    r = w / float(h)  # aspect ratio
    return np.array([[cx], [cy], [s], [r]])

def z_to_xywh(x):
    """Convert Kalman state x = [cx, cy, s, r] to [x, y, w, h]."""
    cx, cy, s, r = x[:4].reshape(-1)
    sr = s * r
    sr = max(sr, 1e-6)    # Ensure non-negative
    w = np.sqrt(sr)
    h = s / max(w, 1e-6)  # Prevent division by 0
    x = cx - w / 2
    y = cy - h / 2
    return np.array([x, y, w, h]).reshape(-1)


class KalmanBoxTracker:
    """
    Single object tracker using a Kalman Filter.

    Process noise (Q): The uncertainty that affects the system state update.
                       It controls the smoothness of the state prediction.

    Measurement noise (R): The error that affects the target observation results.
                           It is initialized by KalmanFilter by default.

    Initial uncertainty (P): Controls the initial uncertainty of the state estimate. A larger value is set to indicate
                             that the target's velocity and position uncertainty are high at the beginning.
    """
    count = 0

    def __init__(self, bbox):
        # Define constant velocity model with 7D state: [cx, cy, s, r, vx, vy, vs]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.F[0, 4] = 1  # cx += vx
        self.kf.F[1, 5] = 1  # cy += vy
        self.kf.F[2, 6] = 1  # s  += vs

        self.kf.H = np.eye(4, 7)  # Measurement: [cx, cy, s, r]

        self.kf.P[4:, 4:] *= 1000.  # High uncertainty for velocities
        self.kf.P *= 10.
        # self.kf.Q *= 0.01           # Process noise

        # Process noise Q: small for position, moderate for velocity
        self.kf.Q = np.eye(7)
        self.kf.Q[0:4, 0:4] *= 0.01
        self.kf.Q[4:, 4:] *= 0.1

        # Initial state
        self.kf.x[:4] = xywh_to_z(bbox)
        self.kf.x[4:] = 0.  # Initial velocities

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.hits = 1
        self.hit_streak = 1
        self.history = []

    def update(self, bbox):
        """Update state with new bounding box."""
        self.kf.update(xywh_to_z(bbox))
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.history = []

    def predict(self):
        """Predict the next state."""
        self.kf.predict()
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        pred_bbox = z_to_xywh(self.kf.x)

        # Handle extreme values
        if np.any(np.isnan(pred_bbox)) or np.any(np.abs(pred_bbox) > 1e5):
            pred_bbox = self.history[-1] if self.history else np.zeros(4)  # fallback

        self.history.append(pred_bbox)
        return pred_bbox

    def get_state(self):
        """Return current bounding box estimate."""
        return z_to_xywh(self.kf.x)
