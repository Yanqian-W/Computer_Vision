import cv2
import numpy as np


def get_color(id):
    # Fixed seed for each ID to ensure consistent color
    np.random.seed(id)
    return tuple(int(c) for c in np.random.randint(0, 255, size=3))

def draw_tracks(frame, tracks):
    for t in tracks:
        x, y, w, h = map(int, (t['bbox']))  # Ensure all values are integers
        track_id = t['id']
        label = t['label']
        color = get_color(track_id)  # Get the color of the ID

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=3)
        # Draw label + track ID
        text = f'{label} ID: {track_id}'
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=2)
