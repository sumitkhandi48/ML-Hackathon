"""
Isolated Server-Side Evaluator Service for PathSnap Trace-Lite
Computes aggregate evaluation metrics against ground-truth reference routes.
STRICT ISOLATION: Never exposes reference routes or pointwise ground-truth to the UI.
"""
from pathlib import Path
from typing import Optional
import numpy as np

from src.data_loader import load_trip
from src.label_alignment import align_reference_route
from src.metrics import evaluate_predictions

class EvaluatorService:
    """
    Isolated evaluation service running server-side.
    Called ONLY AFTER inference completes.
    """
    def __init__(self):
        pass

    def evaluate_trip(
        self,
        trip_id: str,
        predicted_arcs: list[int],
        partition: str = "student_test"
    ) -> Optional[dict]:
        """
        Evaluate predicted arc sequence against reference route.
        Returns aggregate metrics only.
        """
        try:
            trip = load_trip(trip_id, partition)
            if trip.reference_route is None or len(trip.reference_route) == 0:
                return {
                    "evaluation_available": False,
                    "reason": "No reference route available for this trip partition."
                }

            T = len(predicted_arcs)
            pred_arr = np.array(predicted_arcs, dtype=np.int32)

            # Align reference route server-side
            aligned_arcs, _ = align_reference_route(
                trip.track_x, trip.track_y, trip.track_times,
                trip.reference_route,
                trip.x1, trip.y1, trip.x2, trip.y2
            )

            # Route continuity
            continuous_count = 0
            for t in range(T - 1):
                a1 = pred_arr[t]
                a2 = pred_arr[t + 1]
                if a1 != -1 and a2 != -1:
                    if a1 == a2:
                        continuous_count += 1
                    elif trip.arcs_to[a1] == trip.arcs_from[a2]:
                        continuous_count += 1
            continuity = float(continuous_count / (T - 1)) if T > 1 else 1.0

            # Observation Top-1 accuracy
            valid_mask = (aligned_arcs != -1) & (pred_arr != -1)
            correct_count = int(np.sum((pred_arr == aligned_arcs) & valid_mask))
            top1_accuracy = float(correct_count / T) if T > 0 else 0.0

            # Sequence LCS F1
            from src.metrics import compute_lcs_length, compress_sequence
            pred_seq = compress_sequence(pred_arr)
            target_seq = compress_sequence(trip.reference_route)
            lcs_len = compute_lcs_length(pred_seq, target_seq)
            lcs_prec = float(lcs_len / len(pred_seq)) if len(pred_seq) > 0 else 0.0
            lcs_rec = float(lcs_len / len(target_seq)) if len(target_seq) > 0 else 0.0
            lcs_f1 = float(2.0 * lcs_prec * lcs_rec / (lcs_prec + lcs_rec)) if (lcs_prec + lcs_rec) > 0 else 0.0

            return {
                "evaluation_available": True,
                "top1_accuracy_percent": round(top1_accuracy * 100.0, 2),
                "route_continuity_percent": round(continuity * 100.0, 2),
                "sequence_lcs_f1_percent": round(lcs_f1 * 100.0, 2),
                "lcs_matched_arcs": lcs_len,
                "total_observations": T,
                "correct_observations": correct_count
            }
        except Exception as e:
            return {
                "evaluation_available": False,
                "error": str(e)
            }
