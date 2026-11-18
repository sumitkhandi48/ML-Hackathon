"""
Reference-Constrained Label Alignment for PathSnap Trace-Lite
Aligns chronological reference route arcs to pointwise GPS observations
using dynamic programming with monotonic progression and distance emissions.
"""
import numpy as np
from src.geo_utils import point_to_segment_distance

def align_reference_route(
    track_x: np.ndarray,
    track_y: np.ndarray,
    track_times: np.ndarray,
    reference_route: np.ndarray,
    arc_x1: np.ndarray,
    arc_y1: np.ndarray,
    arc_x2: np.ndarray,
    arc_y2: np.ndarray,
    sigma_dist: float = 15.0,
    skip_penalty: float = 0.5,
    max_skip: int = 8
) -> tuple[np.ndarray, dict]:
    """
    Align GPS observations to reference route positions.
    
    Parameters:
        track_x, track_y: GPS coordinates in meters (T,)
        track_times: GPS timestamps in seconds (T,)
        reference_route: Chronological sequence of arc IDs in ground-truth route (L,)
        arc_x1, arc_y1, arc_x2, arc_y2: Segment endpoints for all arcs in the trip graph
        sigma_dist: Gaussian standard deviation for emission distance
        skip_penalty: Cost penalty per skipped route arc
        max_skip: Maximum forward jump along reference route in one step
        
    Returns:
        aligned_arc_ids: Array of shape (T,) with the assigned arc ID for each GPS observation
        stats: Dictionary with alignment statistics (fallback count, ambiguous count, etc.)
    """
    T = len(track_x)
    L = len(reference_route)
    
    if L == 0:
        return np.full(T, -1, dtype=np.int32), {"fallback_count": T, "ambiguous_count": 0, "status": "empty_route"}
    
    # Extract coordinates for reference route arcs
    ref_x1 = arc_x1[reference_route]
    ref_y1 = arc_y1[reference_route]
    ref_x2 = arc_x2[reference_route]
    ref_y2 = arc_y2[reference_route]
    
    # Compute median segment length along reference route to adaptively calibrate transition penalties
    seg_lens = np.hypot(ref_x2 - ref_x1, ref_y2 - ref_y1)
    median_seg_len = float(np.median(seg_lens)) if len(seg_lens) > 0 else 25.0
    # Adapt skip penalty proportionally to physical segment length (micro-segments vs highway spans)
    adaptive_skip_penalty = skip_penalty * min(1.0, max(0.05, median_seg_len / 50.0))

    # Precompute distance matrix of shape (T, L)
    # Since L is typically 100-800 and T is 500-2000, T*L is ~1M floats, easily vectorized
    dist_matrix = np.zeros((T, L), dtype=np.float32)
    for t in range(T):
        dists, _, _, _ = point_to_segment_distance(
            track_x[t], track_y[t],
            ref_x1, ref_y1, ref_x2, ref_y2
        )
        dist_matrix[t] = dists
        
    # DP table: dp[t, k] = max log-likelihood of matching observation t to route position k
    # backpointer[t, k] = best previous route position k' at time t-1
    NEG_INF = -1e9
    dp = np.full((T, L), NEG_INF, dtype=np.float32)
    backpointer = np.full((T, L), -1, dtype=np.int32)
    
    # Emission score: - (dist^2) / (2 * sigma_dist^2)
    inv_2sigma2 = 1.0 / (2.0 * sigma_dist * sigma_dist)
    emission = - (dist_matrix ** 2) * inv_2sigma2
    
    # Initialization at t = 0: allow starting at early positions in route (e.g. first min(15, L))
    init_window = min(15, L)
    for k in range(init_window):
        dp[0, k] = emission[0, k] - k * 0.5
        
    # If initial window has very poor distance, allow closest among all
    if np.max(dp[0]) < -50.0:
        best_k = int(np.argmin(dist_matrix[0]))
        dp[0, best_k] = emission[0, best_k]
        
    # DP forward pass with beam window pruning
    for t in range(1, T):
        dx = track_x[t] - track_x[t-1]
        dy = track_y[t] - track_y[t-1]
        d_gps = float(np.hypot(dx, dy))
        dt = max(1.0, float(track_times[t] - track_times[t-1]))
        
        # Scale max skip with physical displacement and segment granularity
        plausible_skips = int(d_gps / max(3.0, median_seg_len)) + int(dt * 3) + 8
        effective_max_skip = max(max_skip, min(40, plausible_skips))
        
        # Candidate positions around the best previous route position (beam window)
        prev_best_k = int(np.argmax(dp[t-1]))
        if dp[t-1, prev_best_k] <= NEG_INF / 2:
            # Fallback if path broke
            best_k = int(np.argmin(dist_matrix[t]))
            dp[t, best_k] = emission[t, best_k]
            continue
            
        min_k = max(0, prev_best_k - 2)
        max_k = min(L - 1, prev_best_k + effective_max_skip + 3)
        
        for k in range(min_k, max_k + 1):
            # Possible predecessors k_prev <= k within beam
            start_prev = max(min_k, k - effective_max_skip)
            cand_prev = np.arange(start_prev, k + 1)
            
            scores = dp[t-1, cand_prev] - (k - cand_prev) * adaptive_skip_penalty
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            
            if best_score > NEG_INF / 2:
                dp[t, k] = emission[t, k] + best_score
                backpointer[t, k] = cand_prev[best_idx]
                
        # If all dp[t] are -inf, restart from closest position in route
        if np.max(dp[t]) <= NEG_INF / 2:
            subset_dists = dist_matrix[t, prev_best_k:min(L, prev_best_k + 20)]
            if len(subset_dists) > 0 and np.min(subset_dists) <= 50.0:
                chosen_k = prev_best_k + int(np.argmin(subset_dists))
            else:
                chosen_k = int(np.argmin(dist_matrix[t]))
            dp[t, chosen_k] = emission[t, chosen_k]
            backpointer[t, chosen_k] = prev_best_k

    # Backtracking
    aligned_positions = np.zeros(T, dtype=np.int32)
    best_last_k = int(np.argmax(dp[T-1]))
    if dp[T-1, best_last_k] <= NEG_INF / 2:
        best_last_k = int(np.argmin(dist_matrix[T-1]))
        
    aligned_positions[T-1] = best_last_k
    for t in range(T-1, 0, -1):
        prev_k = backpointer[t, aligned_positions[t]]
        if prev_k == -1 or prev_k > aligned_positions[t]:
            # fallback
            prev_k = max(0, aligned_positions[t] - 1)
        aligned_positions[t-1] = prev_k

    # Convert route positions to actual arc IDs
    aligned_arc_ids = reference_route[aligned_positions]
    
    # Compute alignment diagnostics
    # Fallback frequency: points where distance to aligned arc > 35m
    aligned_dists = dist_matrix[np.arange(T), aligned_positions]
    fallback_mask = aligned_dists > 35.0
    fallback_count = int(np.sum(fallback_mask))
    
    # Ambiguous alignments: points where second best route arc is within 3 meters
    sorted_dists = np.sort(dist_matrix, axis=1)
    ambiguous_mask = (sorted_dists[:, 1] - sorted_dists[:, 0]) < 3.0
    ambiguous_count = int(np.sum(ambiguous_mask))
    
    stats = {
        "fallback_count": fallback_count,
        "fallback_fraction": float(fallback_count / T),
        "ambiguous_count": ambiguous_count,
        "ambiguous_fraction": float(ambiguous_count / T),
        "mean_distance": float(np.mean(aligned_dists)),
        "max_distance": float(np.max(aligned_dists)),
        "total_points": T,
        "unique_arcs_matched": len(np.unique(aligned_arc_ids)),
        "reference_route_length": L
    }
    
    return aligned_arc_ids, stats
