"""
Feature Extraction for PathSnap Trace-Lite
Extracts candidate features using ONLY GPS and road-network information.
No raw arc IDs, trip IDs, or reference-dependent features are used.
"""
from dataclasses import dataclass
import numpy as np

from src.data_loader import TripData
from src.candidate_gen import ObservationCandidates
from src.geo_utils import calculate_heading, heading_difference

FEATURE_NAMES = [
    "dist_to_segment",
    "inv_dist",
    "heading_diff",
    "smoothed_gps_heading",
    "heading_reliability",
    "heading_diff_weighted",
    "projection_fraction",
    "segment_length",
    "estimated_speed",
    "timestamp_interval",
    "candidate_distance_rank",
    "dist_to_start",
    "dist_to_end"
]


def smooth_angles(angles: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Smooth circular angles in degrees [0, 360) using vector mean over a sliding window.
    """
    rad = np.radians(angles)
    sin_vals = np.sin(rad)
    cos_vals = np.cos(rad)
    
    # Kernel convolution with reflective padding
    pad = window // 2
    sin_padded = np.pad(sin_vals, pad, mode="edge")
    cos_padded = np.pad(cos_vals, pad, mode="edge")
    
    kernel = np.ones(window) / window
    sin_smooth = np.convolve(sin_padded, kernel, mode="valid")
    cos_smooth = np.convolve(cos_padded, kernel, mode="valid")
    
    smoothed = np.degrees(np.arctan2(sin_smooth, cos_smooth))
    smoothed = (smoothed + 360.0) % 360.0
    return smoothed


def compute_trajectory_motion(trip: TripData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute timestamp intervals, speeds, raw headings, and smoothed headings for GPS points.
    
    Returns:
        dts: (T,) interval in seconds
        speeds: (T,) speed in m/s
        raw_headings: (T,) compass heading in degrees [0, 360)
        smoothed_headings: (T,) smoothed compass heading in degrees [0, 360)
    """
    T = len(trip.track_times)
    dts = np.ones(T, dtype=np.float32)
    speeds = np.zeros(T, dtype=np.float32)
    raw_headings = np.zeros(T, dtype=np.float32)
    
    if T > 1:
        dt = np.diff(trip.track_times)
        # Avoid zero or negative dt
        dt_safe = np.where(dt <= 0.05, 1.0, dt)
        dts[1:] = dt_safe
        dts[0] = dts[1]
        
        dx = np.diff(trip.track_x)
        dy = np.diff(trip.track_y)
        dist = np.hypot(dx, dy)
        
        speed = dist / dt_safe
        speeds[1:] = speed
        speeds[0] = speeds[1]
        
        headings = calculate_heading(dx, dy)
        raw_headings[1:] = headings
        raw_headings[0] = raw_headings[1]
    
    smoothed_headings = smooth_angles(raw_headings, window=5)
    return dts, speeds, raw_headings, smoothed_headings


def extract_candidate_features(
    trip: TripData,
    candidates_list: list[ObservationCandidates]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract candidate feature matrix X, observation index array, candidate arc array, and rank array.
    
    Returns:
        X: (N_cand, num_features) float32
        obs_indices: (N_cand,) int32
        cand_arcs: (N_cand,) int32
        cand_ranks: (N_cand,) int32
    """
    dts, speeds, raw_headings, smoothed_headings = compute_trajectory_motion(trip)
    
    total_candidates = sum(len(c.arc_ids) for c in candidates_list)
    X = np.zeros((total_candidates, len(FEATURE_NAMES)), dtype=np.float32)
    obs_indices = np.zeros(total_candidates, dtype=np.int32)
    cand_arcs = np.zeros(total_candidates, dtype=np.int32)
    cand_ranks = np.zeros(total_candidates, dtype=np.int32)
    
    ptr = 0
    for c in candidates_list:
        t = c.obs_idx
        num_c = len(c.arc_ids)
        if num_c == 0:
            continue
            
        dt = float(dts[t])
        speed = float(speeds[t])
        smooth_head = float(smoothed_headings[t])
        
        # Heading reliability clamped in [0, 1]: 0 when stationary (v < 1 m/s), 1 when v >= 3 m/s
        heading_rel = float(np.clip((speed - 0.5) / 2.5, 0.0, 1.0))
        
        # Segment properties
        c_arcs = c.arc_ids
        seg_heads = trip.arc_headings[c_arcs]
        seg_lens = trip.arc_lengths[c_arcs]
        
        head_diffs = heading_difference(smooth_head, seg_heads)
        head_diffs_weighted = head_diffs * heading_rel
        
        dists = c.distances
        fracs = c.fractions
        ranks = c.ranks
        
        # Populate features
        # 0: dist_to_segment
        X[ptr:ptr+num_c, 0] = dists
        # 1: inv_dist
        X[ptr:ptr+num_c, 1] = 1.0 / (1.0 + dists)
        # 2: heading_diff
        X[ptr:ptr+num_c, 2] = head_diffs
        # 3: smoothed_gps_heading
        X[ptr:ptr+num_c, 3] = smooth_head
        # 4: heading_reliability
        X[ptr:ptr+num_c, 4] = heading_rel
        # 5: heading_diff_weighted
        X[ptr:ptr+num_c, 5] = head_diffs_weighted
        # 6: projection_fraction
        X[ptr:ptr+num_c, 6] = fracs
        # 7: segment_length
        X[ptr:ptr+num_c, 7] = seg_lens
        # 8: estimated_speed
        X[ptr:ptr+num_c, 8] = speed
        # 9: timestamp_interval
        X[ptr:ptr+num_c, 9] = dt
        # 10: candidate_distance_rank
        X[ptr:ptr+num_c, 10] = ranks
        # 11: dist_to_start
        X[ptr:ptr+num_c, 11] = fracs * seg_lens
        # 12: dist_to_end
        X[ptr:ptr+num_c, 12] = (1.0 - fracs) * seg_lens
        
        obs_indices[ptr:ptr+num_c] = t
        cand_arcs[ptr:ptr+num_c] = c_arcs
        cand_ranks[ptr:ptr+num_c] = ranks
        
        ptr += num_c
        
    return X, obs_indices, cand_arcs, cand_ranks
