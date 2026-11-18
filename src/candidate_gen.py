"""
Candidate Generation for PathSnap Trace-Lite
Bounded spatial candidate retrieval using Shapely STRtree,
exact point-to-segment distances, distance ranking, and baseline evaluation.
"""
from dataclasses import dataclass
import numpy as np

from src.config import INITIAL_SEARCH_RADIUS, FALLBACK_SEARCH_RADIUS, TOP_K_CANDIDATES
from src.geo_utils import point_to_segment_distance
from src.data_loader import TripData

@dataclass
class ObservationCandidates:
    """Candidates for a single GPS observation."""
    obs_idx: int
    arc_ids: np.ndarray          # (K,)
    distances: np.ndarray        # (K,) in meters
    fractions: np.ndarray        # (K,) projection fractions in [0, 1]
    proj_x: np.ndarray           # (K,) projected x
    proj_y: np.ndarray           # (K,) projected y
    ranks: np.ndarray            # (K,) 1-based ranks


def generate_trip_candidates(
    trip: TripData,
    k: int = TOP_K_CANDIDATES,
    initial_radius: float = INITIAL_SEARCH_RADIUS,
    fallback_radius: float = FALLBACK_SEARCH_RADIUS
) -> list[ObservationCandidates]:
    """
    Generate bounded candidate arcs for each GPS observation in a trip.
    
    Returns:
        List of ObservationCandidates of length T.
    """
    T = len(trip.track_times)
    candidates_list: list[ObservationCandidates] = []
    
    for t in range(T):
        px = float(trip.track_x[t])
        py = float(trip.track_y[t])
        
        # Initial search
        cand_indices = trip.spatial_index.query_radius(px, py, initial_radius)
        
        # Fallback if too few candidates
        if len(cand_indices) < k:
            cand_indices = trip.spatial_index.query_radius(px, py, fallback_radius)
            
        # Extreme fallback if still empty
        if len(cand_indices) == 0:
            cand_indices = trip.spatial_index.query_radius(px, py, fallback_radius * 2.5)
            
        if len(cand_indices) == 0:
            # Fallback to absolute nearest 1 segment by sampling
            # pick first 100 segments or arbitrary segment
            cand_indices = np.array([0], dtype=np.int32)
            
        cand_indices = np.unique(cand_indices)
        
        # Exact point-to-segment distance
        x1_sub = trip.x1[cand_indices]
        y1_sub = trip.y1[cand_indices]
        x2_sub = trip.x2[cand_indices]
        y2_sub = trip.y2[cand_indices]
        
        dists, frac, proj_x, proj_y = point_to_segment_distance(
            px, py, x1_sub, y1_sub, x2_sub, y2_sub
        )
        
        # Sort by distance
        sort_order = np.argsort(dists)
        top_idx = sort_order[:k]
        
        selected_arc_ids = cand_indices[top_idx]
        selected_dists = dists[top_idx]
        selected_fracs = frac[top_idx]
        selected_px = proj_x[top_idx]
        selected_py = proj_y[top_idx]
        ranks = np.arange(1, len(selected_arc_ids) + 1, dtype=np.int32)
        
        candidates_list.append(ObservationCandidates(
            obs_idx=t,
            arc_ids=selected_arc_ids,
            distances=selected_dists,
            fractions=selected_fracs,
            proj_x=selected_px,
            proj_y=selected_py,
            ranks=ranks
        ))
        
    return candidates_list


def evaluate_candidate_quality(
    candidates_list: list[ObservationCandidates],
    aligned_arc_ids: np.ndarray
) -> dict:
    """
    Evaluate candidate coverage, miss rate, and nearest-road baseline accuracy.
    """
    T = len(aligned_arc_ids)
    covered_count = 0
    nearest_match_count = 0
    rank_positions = []
    
    for t in range(T):
        target = aligned_arc_ids[t]
        cands = candidates_list[t]
        
        # Nearest baseline is rank 1 candidate
        if len(cands.arc_ids) > 0 and cands.arc_ids[0] == target:
            nearest_match_count += 1
            
        # Check if target is in candidates
        match_where = np.where(cands.arc_ids == target)[0]
        if len(match_where) > 0:
            covered_count += 1
            rank_positions.append(int(match_where[0] + 1))
        else:
            rank_positions.append(-1)
            
    coverage_rate = float(covered_count / T) if T > 0 else 0.0
    miss_rate = float(1.0 - coverage_rate)
    top1_accuracy = float(nearest_match_count / T) if T > 0 else 0.0
    
    found_ranks = [r for r in rank_positions if r > 0]
    mean_found_rank = float(np.mean(found_ranks)) if len(found_ranks) > 0 else 0.0
    
    return {
        "total_observations": T,
        "covered_observations": covered_count,
        "candidate_coverage": coverage_rate,
        "candidate_miss_rate": miss_rate,
        "nearest_baseline_top1_accuracy": top1_accuracy,
        "mean_found_rank": mean_found_rank
    }
