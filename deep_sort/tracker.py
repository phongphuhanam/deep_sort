# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from collections import defaultdict
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track, TrackState


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

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3, init_score=0.9):
        self.metric = metric
        if self.metric is not None:
            self._match = self._match_with_features
            self.max_age = max_age
        else:
            self._match = self._match_with_iou
            self.iou_cache = {}
            self.max_age = 1 # since ID is change rapidly, we dont need high max age
        
        self.max_iou_distance = max_iou_distance
        self.n_init = n_init
        self.init_score = init_score

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def get_id_count(self):
        return self._next_id

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        missed_tracks=[]
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            missed_tracks.append(self.tracks[track_idx])
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        if self.metric is not None:
            # Update distance metric.
            active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
            features, targets = [], []
            for track in self.tracks:
                if not track.is_confirmed():
                    continue
                features += track.features
                targets += [track.track_id for _ in track.features]
                track.features = []
            self.metric.partial_fit(
                np.asarray(features), np.asarray(targets), active_targets)
            
        return missed_tracks


    def _match_with_iou(self, detections):
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        
        def iou_metric(tracks, dets, track_indices, detection_indices):
            iou_match = iou_matching.iou_cost2(tracks, dets, track_indices, detection_indices)
            self.iou_cache["id"] = {tracks[k].track_id:idx for idx,k in enumerate(track_indices)}
            self.iou_cache["cost"] = iou_match.copy()
            # if len(self.iou_cache["id"].keys()) == self.iou_cache["cost"].shape[0]:
            #     print()
            return iou_match

        track_indices = np.arange(len(self.tracks)) 
        iou_track_candidates = [
            k for k in track_indices if
            self.tracks[k].time_since_update <= self.max_age]
        
        unmatched_tracks_a  = [
            k for k in track_indices if
            self.tracks[k].time_since_update > self.max_age]

        unmatched_detections = np.arange(len(detections))
        matches, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_metric, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections


    def _match_with_features(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        track = Track(mean, covariance, self._next_id, self.n_init, self.max_age, detection)
        if detection.confidence >= self.init_score:
            track.state = TrackState.Confirmed

        self.tracks.append(track)
        self._next_id += 1
