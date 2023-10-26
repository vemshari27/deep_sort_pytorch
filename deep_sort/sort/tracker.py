# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, metric_l, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.metric_l = metric_l
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections, transform_pipeline):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections, transform_pipeline)
        
        # if len(detections)>0: #extra
        #     if len(self.tracks) > 0:
        #         matches = [[0,0]]
        #         unmatched_tracks = []
        #         unmatched_detections = []

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, features_l, targets = [], [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            features_l += track.features_l
            targets += [track.track_id for _ in track.features]
            track.features = []
            track.features_l = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)
        self.metric_l.partial_fit(
            np.asarray(features_l), np.asarray(targets), active_targets)
        
        # print("No of trackers", len(self.tracks)) #extra
        # for i in self.tracks:
        #     print(i)

    def _match(self, detections, transform_pipeline):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            # print("Cost Matrix", cost_matrix)

            return cost_matrix
        
        def gated_metric_l(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature_l for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric_l.distance(features, targets)
            # cost_matrix = linear_assignment.gate_cost_matrix(
            #     self.kf, cost_matrix, tracks, dets, track_indices,
            #     detection_indices)
            # print("Cost Matrix l", cost_matrix)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        
        # print("No of confirmed trackers", len(confirmed_tracks)) #extra
        # # for i in confirmed_tracks:
        # #     print(self.tracks[i])
        # print("No of unconfirmed trackers", len(unconfirmed_tracks)) #extra
        # # for i in unconfirmed_tracks:
        # #     print(self.tracks[i])

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)
        
        # print("Appearance matches", matches_a, unmatched_tracks_a, unmatched_detections)
        
        matches_c, unmatched_tracks_c, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric_l, self.metric_l.matching_threshold, self.max_age,
                self.tracks, detections, unmatched_tracks_a, unmatched_detections)
        
        # print("L matches", matches_c, unmatched_tracks_c, unmatched_detections)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_c if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_c = [
            k for k in unmatched_tracks_c if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b + matches_c
        unmatched_tracks = list(set(unmatched_tracks_c + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

        # # Associate remaining tracks together with unconfirmed tracks using IOU.
        # iou_track_candidates = unconfirmed_tracks + [
        #     k for k in unmatched_tracks_a if
        #     self.tracks[k].time_since_update == 1]
        # unmatched_tracks_a = [
        #     k for k in unmatched_tracks_a if
        #     self.tracks[k].time_since_update != 1]
        # matches_b, unmatched_tracks_b, unmatched_detections = \
        #     linear_assignment.min_cost_matching(
        #         iou_matching.iou_cost, self.max_iou_distance, self.tracks,
        #         detections, iou_track_candidates, unmatched_detections)

        # matches = matches_a + matches_b
        # unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        # return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature, detection.feature_l, detection.depth))
        self._next_id += 1
