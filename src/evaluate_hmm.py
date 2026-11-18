"""
HMM Route Reconstruction and Evaluation for PathSnap Trace-Lite
Evaluates Classifier-Only vs Classifier + HMM / Viterbi decoding
across all ten student-testing trips.
Records pointwise accuracy, route continuity, sequence LCS F1, and failure analysis.
"""
import json
import time
from pathlib import Path
import joblib
import numpy as np

from src.config import ARTIFACTS_DIR, EXPERIMENTS_DIR
from src.data_loader import load_trip, list_trips
from src.candidate_gen import generate_trip_candidates
from src.features import extract_candidate_features
from src.viterbi import ViterbiRouteDecoder
from src.metrics import evaluate_predictions

def run_hmm_evaluation():
    """
    Evaluate Classifier-Only vs Classifier+HMM on all 10 student-test trips.
    """
    # Load model: prefer final tuned model if exists, else best classifier
    tuned_model_path = ARTIFACTS_DIR / "final_tuned_model.joblib"
    best_model_path = ARTIFACTS_DIR / "best_classifier.joblib"
    
    model_path = tuned_model_path if tuned_model_path.exists() else best_model_path
    if not model_path.exists():
        raise FileNotFoundError("No trained model found. Run train_eval.py first.")
        
    print(f"Loading model artifact from {model_path}...", flush=True)
    clf = joblib.load(model_path)
    
    test_trips = list_trips("student_test")
    print(f"Running HMM evaluation on {len(test_trips)} student-test trips...", flush=True)
    
    trip_results = []
    
    total_clf_time = 0.0
    total_hmm_time = 0.0
    
    for tid in test_trips:
        trip = load_trip(tid, "student_test")
        candidates = generate_trip_candidates(trip, k=8)
        X, obs_indices, cand_arcs, cand_ranks = extract_candidate_features(trip, candidates)
        
        # 1. Classifier-Only Prediction
        t_clf_0 = time.time()
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)[:, 1]
        else:
            scores = clf.decision_function(X)
            probs = 1.0 / (1.0 + np.exp(-scores))
            
        T = len(trip.track_times)
        clf_pred_arcs = np.full(T, -1, dtype=np.int32)
        cand_probs_per_obs = []
        
        for t in range(T):
            mask = obs_indices == t
            if np.any(mask):
                c_p = probs[mask]
                c_a = cand_arcs[mask]
                best_idx = np.argmax(c_p)
                clf_pred_arcs[t] = c_a[best_idx]
                cand_probs_per_obs.append(c_p)
            else:
                cand_probs_per_obs.append(np.array([], dtype=np.float32))
                
        clf_time = time.time() - t_clf_0
        total_clf_time += clf_time
        
        # 2. Classifier + HMM / Viterbi Decoding
        t_hmm_0 = time.time()
        decoder = ViterbiRouteDecoder(trip, beta=20.0, max_speed=35.0)
        hmm_pred_arcs, reconstructed_route, viterbi_diag = decoder.decode(candidates, cand_probs_per_obs)
        hmm_time = time.time() - t_hmm_0
        total_hmm_time += (clf_time + hmm_time)
        
        # Ground truth reference alignment for student test trip
        # To evaluate on test data, align route if .route file exists
        if trip.reference_route is not None:
            from src.label_alignment import align_reference_route
            ref_aligned, _ = align_reference_route(
                trip.track_x, trip.track_y, trip.track_times,
                trip.reference_route,
                trip.x1, trip.y1, trip.x2, trip.y2
            )
            cand_cov = np.array([ref_aligned[t] in candidates[t].arc_ids for t in range(T)], dtype=bool)
            
            # Ground truth for candidates
            target_per_cand = ref_aligned[obs_indices]
            cand_y = (cand_arcs == target_per_cand).astype(int)
            
            # Evaluate Classifier-Only
            clf_metrics = evaluate_predictions(
                ref_aligned, clf_pred_arcs,
                cand_y, probs, cand_cov,
                trip.arcs_from, trip.arcs_to,
                trip.reference_route
            )
            
            # Evaluate Classifier + HMM
            hmm_metrics = evaluate_predictions(
                ref_aligned, hmm_pred_arcs,
                cand_y, probs, cand_cov,
                trip.arcs_from, trip.arcs_to,
                trip.reference_route
            )
            
            # Failure and improvement analysis
            improved_points = int(np.sum((clf_pred_arcs != ref_aligned) & (hmm_pred_arcs == ref_aligned)))
            degraded_points = int(np.sum((clf_pred_arcs == ref_aligned) & (hmm_pred_arcs != ref_aligned)))
            
            trip_stat = {
                "trip_id": tid,
                "num_observations": T,
                "clf_top1": clf_metrics["top1_accuracy"],
                "hmm_top1": hmm_metrics["top1_accuracy"],
                "top1_delta": hmm_metrics["top1_accuracy"] - clf_metrics["top1_accuracy"],
                "clf_continuity": clf_metrics["route_continuity"],
                "hmm_continuity": hmm_metrics["route_continuity"],
                "continuity_delta": hmm_metrics["route_continuity"] - clf_metrics["route_continuity"],
                "clf_lcs_f1": clf_metrics["lcs_f1"],
                "hmm_lcs_f1": hmm_metrics["lcs_f1"],
                "lcs_delta": hmm_metrics["lcs_f1"] - clf_metrics["lcs_f1"],
                "improved_points": improved_points,
                "degraded_points": degraded_points,
                "restarts_count": viterbi_diag["restarts_count"],
                "reconstructed_arcs": viterbi_diag["reconstructed_arcs_count"],
                "clf_time_sec": clf_time,
                "hmm_total_time_sec": clf_time + hmm_time
            }
            trip_results.append(trip_stat)
            print(f"  Trip {tid} (N={T}): Top-1 {clf_metrics['top1_accuracy']:.4f} -> {hmm_metrics['top1_accuracy']:.4f} | Cont: {clf_metrics['route_continuity']:.4f} -> {hmm_metrics['route_continuity']:.4f} | LCS: {clf_metrics['lcs_f1']:.4f} -> {hmm_metrics['lcs_f1']:.4f} | Restarts: {viterbi_diag['restarts_count']}", flush=True)

    # Aggregated results
    mean_clf_top1 = float(np.mean([r["clf_top1"] for r in trip_results]))
    mean_hmm_top1 = float(np.mean([r["hmm_top1"] for r in trip_results]))
    mean_clf_cont = float(np.mean([r["clf_continuity"] for r in trip_results]))
    mean_hmm_cont = float(np.mean([r["hmm_continuity"] for r in trip_results]))
    mean_clf_lcs = float(np.mean([r["clf_lcs_f1"] for r in trip_results]))
    mean_hmm_lcs = float(np.mean([r["hmm_lcs_f1"] for r in trip_results]))
    
    overall_summary = {
        "num_test_trips": len(trip_results),
        "classifier_only": {
            "mean_top1_accuracy": mean_clf_top1,
            "mean_route_continuity": mean_clf_cont,
            "mean_lcs_f1": mean_clf_lcs,
            "total_inference_time_sec": total_clf_time
        },
        "classifier_plus_hmm": {
            "mean_top1_accuracy": mean_hmm_top1,
            "mean_route_continuity": mean_hmm_cont,
            "mean_lcs_f1": mean_hmm_lcs,
            "total_inference_time_sec": total_hmm_time
        },
        "improvements": {
            "top1_accuracy_change": mean_hmm_top1 - mean_clf_top1,
            "route_continuity_change": mean_hmm_cont - mean_clf_cont,
            "lcs_f1_change": mean_hmm_lcs - mean_clf_lcs
        },
        "trip_level_results": trip_results
    }
    
    out_path = EXPERIMENTS_DIR / "hmm_evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(overall_summary, f, indent=2)
    print(f"\nSaved HMM evaluation results to {out_path}", flush=True)
    
    print("\n--- CLASSIFIER VS HMM SUMMARY ---")
    print(f"Top-1 Accuracy:   {mean_clf_top1:.4f} --> {mean_hmm_top1:.4f} (Δ = {mean_hmm_top1 - mean_clf_top1:+.4f})")
    print(f"Route Continuity: {mean_clf_cont:.4f} --> {mean_hmm_cont:.4f} (Δ = {mean_hmm_cont - mean_clf_cont:+.4f})")
    print(f"Sequence LCS F1:  {mean_clf_lcs:.4f} --> {mean_hmm_lcs:.4f} (Δ = {mean_hmm_lcs - mean_clf_lcs:+.4f})")
    print(f"Total Time:       {total_clf_time:.2f}s --> {total_hmm_time:.2f}s")
    
    return overall_summary


if __name__ == "__main__":
    run_hmm_evaluation()
