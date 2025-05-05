from tracking.tracker.kalman_filter import KalmanBoxTracker


def _compute_iou(bbox1, bbox2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    :param bbox1: First bounding box as (x, y, w, h)
    :param bbox2: Second bounding box as (x, y, w, h)
    :return: IoU value
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    # Compute intersection area
    x1_int = max(x1, x2)
    y1_int = max(y1, y2)
    x2_int = min(x1 + w1, x2 + w2)
    y2_int = min(y1 + h1, y2 + h2)

    intersection_area = max(0, x2_int - x1_int) * max(0, y2_int - y1_int)
    # Compute union area
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    return intersection_area / union_area if union_area > 0 else 0


class SORT:
    """
    Simple Online and Realtime Tracking (SORT) algorithm.
    Uses Kalman Filter and IoU-based matching for object tracking.
    """

    def __init__(self, iou_threshold=0.3, min_hits=3):
        """
        Initializes the SORT tracker.

        :param iou_threshold: The threshold for the Intersection over Union (IoU) matching.
        """
        self.trackers = {}  # Keeps track of the object trackers by their ID
        self.iou_threshold = iou_threshold  # IoU threshold for matching detections to existing trackers
        self.min_hits = min_hits
        self.next_id = 0  # The next available ID to assign to a new tracker

    def update(self, detections, frame=None, reid=None):
        updated_tracks = []
        unmatched_detections = detections.copy()
        unmatched_tracks = list(self.trackers.keys())

        predicted_bboxes = {tid: tracker.predict() for tid, tracker in self.trackers.items()}

        # Step 1: IoU-based matching
        matched, unmatched_detections, unmatched_tracks = self._iou_matching(predicted_bboxes, unmatched_detections,
                                                                             unmatched_tracks)

        # Step 2: ReID-based matching (optional)
        if reid is not None and frame is not None:
            reid_matched, unmatched_detections, unmatched_tracks = self._reid_matching(
                reid, frame, unmatched_detections, unmatched_tracks, predicted_bboxes
            )
            matched.update(reid_matched)

        # Step 3: Update matched trackers
        for track_id, det in matched.items():
            self.trackers[track_id].update(det['bbox'])  # hit_streak handled inside
            updated_tracks.append({
                'id': track_id,
                'bbox': self.trackers[track_id].get_state(),
                'label': det['label']
            })

        # Step 4: Update unmatched trackers (reset hit_streak)
        for track_id in unmatched_tracks:
            self.trackers[track_id].hit_streak = 0

        # Step 5: Create new trackers
        for det in unmatched_detections:
            new_tracker = KalmanBoxTracker(det['bbox'])
            self.trackers[self.next_id] = new_tracker
            if self.min_hits <= 1:
                updated_tracks.append({
                    'id': self.next_id,
                    'bbox': new_tracker.get_state(),
                    'label': det['label']
                })
            self.next_id += 1

        # Step 6: Filter by min_hits
        final_tracks = []
        for track in updated_tracks:
            tid = track['id']
            if self.trackers[tid].hit_streak >= self.min_hits:
                final_tracks.append(track)

        return final_tracks

    def _iou_matching(self, predicted_bboxes, detections, track_ids):
        """
        Match detections to trackers based on IoU.

        :param predicted_bboxes: List of predicted bounding boxes from existing trackers
        :param detections: List of new detections with bounding boxes and labels
        :return: A dictionary of matched trackers and a list of unmatched detections
        """
        matched = {}
        unmatched_detections = []
        unmatched_tracks = track_ids.copy()
        used_track_ids = set()

        for det in detections:
            best_iou, best_tid = 0, None
            for tid in track_ids:
                if tid in used_track_ids:
                    continue
                iou = _compute_iou(predicted_bboxes[tid], det['bbox'])
                if iou > best_iou:
                    best_iou, best_tid = iou, tid

            if best_iou > self.iou_threshold:
                matched[best_tid] = det
                used_track_ids.add(best_tid)
                if best_tid in unmatched_tracks:
                    unmatched_tracks.remove(best_tid)
            else:
                unmatched_detections.append(det)

        return matched, unmatched_detections, unmatched_tracks

    def _reid_matching(self, reid, frame, detections, track_ids, predicted_bboxes):
        """
        Match detections to trackers based on ReID features.

        :param predicted_bboxes: Dict of predicted bounding boxes from existing trackers
        :param detections: List of new detections with bounding boxes and labels
        :return: A dict of matched trackers and lists of unmatched detections and tracks
        """
        matched = {}
        unmatched_detections = detections.copy()
        unmatched_tracks = track_ids.copy()

        # Extract detection features
        det_feats = []
        for det in detections:
            feat = reid.extract_feature(frame, det['bbox'])
            if feat is not None:
                det_feats.append((det, feat))

        # Extract track features
        track_feats = []
        for tid in track_ids:
            bbox = predicted_bboxes[tid]
            feat = reid.extract_feature(frame, bbox)
            if feat is not None:
                track_feats.append((tid, feat))

        for det, d_feat in det_feats:
            best_sim, best_tid = -1, None
            for tid, t_feat in track_feats:
                sim = reid.cosine_similarity(d_feat, t_feat)
                if sim > best_sim:
                    best_sim, best_tid = sim, tid

            if best_sim > reid.get_similarity_threshold() and best_tid in unmatched_tracks:
                matched[best_tid] = det
                unmatched_detections.remove(det)
                unmatched_tracks.remove(best_tid)

        return matched, unmatched_detections, unmatched_tracks


