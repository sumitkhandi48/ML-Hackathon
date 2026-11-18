"""
Feature Extraction and Caching Pipeline for PathSnap Trace-Lite
Processes trips one by one, aligns reference labels, generates candidates,
extracts features, and caches per-trip npz files in artifacts/cache/.
"""
from pathlib import Path
import time
import numpy as np

from src.config import ARTIFACTS_DIR, TOP_K_CANDIDATES
from src.data_loader import load_trip, list_trips
from src.label_alignment import align_reference_route
from src.candidate_gen import generate_trip_candidates
from src.features import extract_candidate_features, FEATURE_NAMES

CACHE_DIR = ARTIFACTS_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def process_trip_features(trip_id: str, partition: str = "train") -> dict:
    """
    Process a single trip: alignment (if train/route available), candidate generation, feature extraction.
    Saves to CACHE_DIR / f"{partition}_{trip_id}.npz".
    """
    cache_file = CACHE_DIR / f"{partition}_{trip_id}.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        return {
            "trip_id": trip_id,
            "partition": partition,
            "num_candidates": len(data["y"]),
            "num_obs": len(data["aligned_target"]),
            "from_cache": True
        }

    t0 = time.time()
    trip = load_trip(trip_id, partition)
    T = len(trip.track_times)
    
    # Label alignment if reference route exists
    if trip.reference_route is not None:
        aligned_arcs, align_stats = align_reference_route(
            trip.track_x, trip.track_y, trip.track_times,
            trip.reference_route,
            trip.x1, trip.y1, trip.x2, trip.y2
        )
    else:
        aligned_arcs = np.full(T, -1, dtype=np.int32)
        align_stats = {}

    # Candidate generation
    candidates = generate_trip_candidates(trip, k=TOP_K_CANDIDATES)

    # Feature extraction
    X, obs_indices, cand_arcs, cand_ranks = extract_candidate_features(trip, candidates)

    # Ground truth labels for candidates
    # y = 1 if candidate arc == aligned_arcs[obs_idx], else 0
    target_per_cand = aligned_arcs[obs_indices]
    y = ((cand_arcs == target_per_cand) & (target_per_cand != -1)).astype(np.int32)

    # Coverage mask per observation
    cand_coverage = np.zeros(T, dtype=bool)
    for t in range(T):
        target = aligned_arcs[t]
        c = candidates[t]
        if target != -1 and np.any(c.arc_ids == target):
            cand_coverage[t] = True

    # Save to disk
    np.savez_compressed(
        cache_file,
        X=X.astype(np.float32),
        y=y.astype(np.int32),
        obs_indices=obs_indices.astype(np.int32),
        cand_arcs=cand_arcs.astype(np.int32),
        cand_ranks=cand_ranks.astype(np.int32),
        aligned_target=aligned_arcs.astype(np.int32),
        cand_coverage=cand_coverage.astype(bool),
        arcs_from=trip.arcs_from.astype(np.int32),
        arcs_to=trip.arcs_to.astype(np.int32),
        reference_route=trip.reference_route if trip.reference_route is not None else np.array([], dtype=np.int32)
    )
    
    elapsed = time.time() - t0
    return {
        "trip_id": trip_id,
        "partition": partition,
        "num_candidates": len(y),
        "num_obs": T,
        "positive_candidates": int(np.sum(y)),
        "coverage_rate": float(np.mean(cand_coverage)),
        "elapsed_sec": elapsed,
        "from_cache": False
    }


def build_all_features(smoke_test: bool = False, max_trips: int = 5):
    """
    Build cache for all training and test trips.
    """
    train_trips = list_trips("train")
    test_trips = list_trips("student_test")

    if smoke_test:
        train_trips = train_trips[:max_trips]
        test_trips = test_trips[:max_trips]

    print(f"Building features: {len(train_trips)} train trips, {len(test_trips)} test trips...", flush=True)
    train_results = []
    for tid in train_trips:
        res = process_trip_features(tid, "train")
        train_results.append(res)
        print(f"  Train {tid}: {res['num_obs']} obs, {res['num_candidates']} cands, cov={res.get('coverage_rate', 0):.3f}", flush=True)

    test_results = []
    for tid in test_trips:
        res = process_trip_features(tid, "student_test")
        test_results.append(res)
        print(f"  Test {tid}: {res['num_obs']} obs, {res['num_candidates']} cands, cov={res.get('coverage_rate', 0):.3f}", flush=True)

    return train_results, test_results


if __name__ == "__main__":
    build_all_features()
