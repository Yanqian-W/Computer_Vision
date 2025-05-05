import numpy as np
import cv2
from skimage.feature import hog


class ReIDModule:
    def __init__(self, similarity_threshold=0.3):
        self.similarity_threshold = similarity_threshold  # IoU threshold for matching detections to existing trackers

    def extract_feature(self, frame, bbox):
        x, y, w, h = map(int, bbox)
        H, W = frame.shape[:2]

        # Clip bbox to stay within image boundaries
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))  # Ensure at least width 1 and within right bound
        h = max(1, min(h, H - y))  # Ensure at least height 1 and within bottom bound

        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            return np.zeros(34500)
        # Resize for consistency
        crop = cv2.resize(crop, (64, 128))

        # Convert crop to HSV
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # HSV histogram: H(30 bins), S(32), V(32)
        hist = cv2.calcHist([hsv_crop], [0, 1, 2], None, [30, 32, 32],
                            [0, 180, 0, 256, 0, 256]).flatten()
        hist = hist / (np.linalg.norm(hist) + 1e-6)

        # HOG descriptor
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
        hog_feat = hog_feat / (np.linalg.norm(hog_feat) + 1e-6)

        # Concatenate both features
        feature = np.concatenate([hist, hog_feat])
        return feature

    def cosine_similarity(self, a, b):
        # Ensure a and b are of the same length
        if a.shape[0] != b.shape[0]:
            raise ValueError(f"Feature vectors must have the same dimension. Got {a.shape[0]} and {b.shape[0]}")
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)

    def get_similarity_threshold(self):
        return self.similarity_threshold
