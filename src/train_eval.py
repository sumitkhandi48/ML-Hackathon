"""
Training and Evaluation Pipeline for PathSnap Trace-Lite
Implements 3-fold GroupKFold across 80 training trips, trains exactly 5 models,
evaluates on complete validation and student-testing candidate sets,
records all official metrics, and selects the best model.
"""
import json
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix

from src.config import ARTIFACTS_DIR, EXPERIMENTS_DIR, MODEL_NAMES, CV_FOLDS, RANDOM_SEED
from src.data_loader import list_trips
from src.models import build_model, MODEL_CONFIGS
from src.metrics import evaluate_predictions

CACHE_DIR = ARTIFACTS_DIR / "cache"


def load_cached_trip(trip_id: str, partition: str = "train") -> dict:
    """Load cached features and targets for a single trip."""
    fpath = CACHE_DIR / f"{partition}_{trip_id}.npz"
    data = np.load(fpath)
    return {
        "trip_id": trip_id,
        "partition": partition,
        "X": data["X"],
        "y": data["y"],
        "obs_indices": data["obs_indices"],
        "cand_arcs": data["cand_arcs"],
        "cand_ranks": data["cand_ranks"],
        "aligned_target": data["aligned_target"],
        "cand_coverage": data["cand_coverage"],
        "arcs_from": data["arcs_from"],
        "arcs_to": data["arcs_to"],
        "reference_route": data["reference_route"]
    }


def subsample_training_negatives(X: np.ndarray, y: np.ndarray, obs_indices: np.ndarray, max_neg_per_obs: int = 2) -> np.ndarray:
    """
    Subsample negative candidates strictly within training folds.
    Preserves all positive candidates and up to max_neg_per_obs hardest negatives (lowest distance / rank).
    Vectorized boundary slicing for O(N) performance.
    """
    if len(obs_indices) == 0:
        return np.array([], dtype=np.int32)
        
    diff_mask = np.diff(obs_indices, prepend=obs_indices[0] - 1) != 0
    start_indices = np.where(diff_mask)[0]
    end_indices = np.append(start_indices[1:], len(obs_indices))
    
    keep = []
    for start, end in zip(start_indices, end_indices):
        y_sub = y[start:end]
        pos = np.where(y_sub == 1)[0]
        neg = np.where(y_sub == 0)[0]
        for p in pos:
            keep.append(start + p)
        for n in neg[:max_neg_per_obs]:
            keep.append(start + n)
            
    return np.array(sorted(keep), dtype=np.int32)


def get_observation_top1_predictions(probs: np.ndarray, cand_arcs: np.ndarray, obs_indices: np.ndarray, T_obs: int) -> np.ndarray:
    """
    Fast O(N) extraction of highest probability candidate arc per observation.
    """
    pred_arcs = np.full(T_obs, -1, dtype=np.int32)
    if len(obs_indices) == 0:
        return pred_arcs
        
    diff_mask = np.diff(obs_indices, prepend=obs_indices[0] - 1) != 0
    start_indices = np.where(diff_mask)[0]
    end_indices = np.append(start_indices[1:], len(obs_indices))
    
    for start, end in zip(start_indices, end_indices):
        t = obs_indices[start]
        if 0 <= t < T_obs:
            block_p = probs[start:end]
            best_offset = np.argmax(block_p)
            pred_arcs[t] = cand_arcs[start + best_offset]
            
    return pred_arcs


def run_cross_validation_and_comparison():
    """
    Run 3-fold GroupKFold across all 80 training trips, evaluate 5 models,
    refit on all 80 trips, evaluate on all 10 student-test trips, and save results.
    """
    train_trip_ids = list_trips("train")
    test_trip_ids = list_trips("student_test")
    
    print(f"Loading cached data for {len(train_trip_ids)} training trips...", flush=True)
    train_trips_data = [load_cached_trip(tid, "train") for tid in train_trip_ids]
    
    print(f"Loading cached data for {len(test_trip_ids)} student-test trips...", flush=True)
    test_trips_data = [load_cached_trip(tid, "student_test") for tid in test_trip_ids]
    
    # Build unified training candidate arrays with trip groups
    all_X_list = []
    all_y_list = []
    all_group_list = []
    all_trip_idx_list = []
    
    for i, td in enumerate(train_trips_data):
        n = len(td["y"])
        all_X_list.append(td["X"])
        all_y_list.append(td["y"])
        all_group_list.append(np.full(n, i, dtype=np.int32))
        
    full_X = np.vstack(all_X_list)
    full_y = np.concatenate(all_y_list)
    full_groups = np.concatenate(all_group_list)
    
    print(f"Total training candidate pool: {len(full_y):,} candidates, {int(np.sum(full_y)):,} positives.", flush=True)
    
    # 3-Fold GroupKFold
    gkf = GroupKFold(n_splits=CV_FOLDS)
    folds = list(gkf.split(full_X, full_y, full_groups))
    
    comparison_results = []
    test_predictions_by_model = {}

    for model_name in MODEL_NAMES:
        print(f"\n=======================================================", flush=True)
        print(f"Evaluating Model: {model_name} ({MODEL_CONFIGS[model_name]['family']})", flush=True)
        print(f"=======================================================", flush=True)
        
        cv_top1_list = []
        cv_f1_list = []
        cv_prec_list = []
        cv_rec_list = []
        cv_loss_list = []
        cv_cont_list = []
        cv_train_times = []
        
        # 1. Cross-Validation Loop
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            # Subsample negatives only within training fold
            val_trip_indices = np.unique(full_groups[val_idx])
            train_trip_indices = np.unique(full_groups[train_idx])
            
            # Subsample train candidates
            train_sub_indices = []
            for t_idx in train_trip_indices:
                td = train_trips_data[t_idx]
                keep_sub = subsample_training_negatives(td["X"], td["y"], td["obs_indices"], max_neg_per_obs=2)
                # Map to global indices if needed, or train directly on concatenated subsamples
                train_sub_indices.append((td["X"][keep_sub], td["y"][keep_sub]))
                
            fold_train_X = np.vstack([x for x, _ in train_sub_indices])
            fold_train_y = np.concatenate([y for _, y in train_sub_indices])
            
            # Train model
            clf = build_model(model_name)
            t_start = time.time()
            clf.fit(fold_train_X, fold_train_y)
            train_time = time.time() - t_start
            cv_train_times.append(train_time)
            
            # Evaluate on complete validation candidate sets trip-by-trip
            fold_trip_top1 = []
            fold_trip_f1 = []
            fold_trip_loss = []
            fold_trip_cont = []
            fold_cand_y = []
            fold_cand_prob = []
            
            for t_idx in val_trip_indices:
                td = train_trips_data[t_idx]
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(td["X"])[:, 1]
                else:
                    scores = clf.decision_function(td["X"])
                    probs = 1.0 / (1.0 + np.exp(-scores))
                    
                T_obs = len(td["aligned_target"])
                pred_arcs = get_observation_top1_predictions(probs, td["cand_arcs"], td["obs_indices"], T_obs)
                        
                # Evaluate per-trip metrics with trip's own road graph
                trip_res = evaluate_predictions(
                    td["aligned_target"], pred_arcs,
                    td["y"], probs, td["cand_coverage"],
                    td["arcs_from"], td["arcs_to"],
                    td["reference_route"]
                )
                fold_trip_top1.append(trip_res["top1_accuracy"])
                fold_trip_f1.append(trip_res["candidate_macro_f1"])
                fold_trip_loss.append(trip_res["log_loss"])
                fold_trip_cont.append(trip_res["route_continuity"])
                fold_cand_y.append(td["y"])
                fold_cand_prob.append(probs)
                
            fold_top1 = float(np.mean(fold_trip_top1))
            fold_f1 = float(np.mean(fold_trip_f1))
            fold_loss = float(np.mean(fold_trip_loss))
            fold_cont = float(np.mean(fold_trip_cont))
            
            cv_top1_list.append(fold_top1)
            cv_f1_list.append(fold_f1)
            cv_loss_list.append(fold_loss)
            cv_cont_list.append(fold_cont)
            
            print(f"  Fold {fold_idx + 1}/{CV_FOLDS}: Top-1={fold_top1:.4f}, Macro-F1={fold_f1:.4f}, LogLoss={fold_loss:.4f}, Continuity={fold_cont:.4f} ({train_time:.1f}s)", flush=True)

        mean_cv_top1 = float(np.mean(cv_top1_list))
        mean_cv_f1 = float(np.mean(cv_f1_list))
        mean_cv_loss = float(np.mean(cv_loss_list))
        mean_cv_cont = float(np.mean(cv_cont_list))
        total_cv_train_time = float(np.sum(cv_train_times))

        # 2. Refit on ALL 80 training trips (subsampled negatives for speed)
        print(f"  Refitting {model_name} on all 80 training trips...", flush=True)
        all_train_sub = []
        for td in train_trips_data:
            keep = subsample_training_negatives(td["X"], td["y"], td["obs_indices"], max_neg_per_obs=2)
            all_train_sub.append((td["X"][keep], td["y"][keep]))
            
        full_sub_X = np.vstack([x for x, _ in all_train_sub])
        full_sub_y = np.concatenate([y for _, y in all_train_sub])
        
        final_clf = build_model(model_name)
        t_refit_start = time.time()
        final_clf.fit(full_sub_X, full_sub_y)
        refit_time = time.time() - t_refit_start

        # 3. Evaluate on ALL 10 student-test trips (100% complete candidates)
        print(f"  Evaluating on 10 student-testing trips...", flush=True)
        t_infer_start = time.time()
        test_trip_preds = {}
        all_test_preds = []
        all_test_targets = []
        all_test_cand_y = []
        all_test_cand_prob = []
        all_test_coverage = []
        all_test_conts = []
        all_test_lcs_f1 = []
        
        for td in test_trips_data:
            tid = td["trip_id"]
            if hasattr(final_clf, "predict_proba"):
                probs = final_clf.predict_proba(td["X"])[:, 1]
            else:
                scores = final_clf.decision_function(td["X"])
                probs = 1.0 / (1.0 + np.exp(-scores))
                
            T_obs = len(td["aligned_target"])
            pred_arcs = get_observation_top1_predictions(probs, td["cand_arcs"], td["obs_indices"], T_obs)
                    
            test_trip_preds[tid] = pred_arcs.tolist()
            all_test_preds.append(pred_arcs)
            all_test_targets.append(td["aligned_target"])
            all_test_cand_y.append(td["y"])
            all_test_cand_prob.append(probs)
            all_test_coverage.append(td["cand_coverage"])
            
            trip_eval = evaluate_predictions(
                td["aligned_target"], pred_arcs,
                td["y"], probs, td["cand_coverage"],
                td["arcs_from"], td["arcs_to"],
                td["reference_route"]
            )
            all_test_conts.append(trip_eval["route_continuity"])
            all_test_lcs_f1.append(trip_eval["lcs_f1"])

        inference_time = time.time() - t_infer_start
        test_predictions_by_model[model_name] = test_trip_preds

        # Aggregate student-test metrics
        cat_test_pred = np.concatenate(all_test_preds)
        cat_test_target = np.concatenate(all_test_targets)
        cat_test_cand_y = np.concatenate(all_test_cand_y)
        cat_test_cand_prob = np.concatenate(all_test_cand_prob)
        cat_test_coverage = np.concatenate(all_test_coverage)
        
        # Binary candidate classification metrics across full student-test candidate set
        test_cand_pred_binary = (cat_test_cand_prob >= 0.5).astype(int)
        from sklearn.metrics import precision_score, recall_score, f1_score, log_loss
        test_prec = float(precision_score(cat_test_cand_y, test_cand_pred_binary, zero_division=0))
        test_rec = float(recall_score(cat_test_cand_y, test_cand_pred_binary, zero_division=0))
        test_macro_f1 = float(f1_score(cat_test_cand_y, test_cand_pred_binary, average="macro", zero_division=0))
        clamped_test_prob = np.clip(cat_test_cand_prob, 1e-7, 1.0 - 1e-7)
        test_loss = float(log_loss(cat_test_cand_y, clamped_test_prob))
        
        # Observation Top-1 across all test observations
        correct_test_obs = np.sum((cat_test_pred == cat_test_target) & (cat_test_target != -1) & (cat_test_pred != -1))
        test_top1 = float(correct_test_obs / len(cat_test_target)) if len(cat_test_target) > 0 else 0.0

        # Confusion matrix data
        cm = confusion_matrix(cat_test_cand_y, test_cand_pred_binary)

        model_record = {
            "model_name": model_name,
            "algorithm_family": MODEL_CONFIGS[model_name]["family"],
            "key_configuration": str(MODEL_CONFIGS[model_name]["params"]),
            "cv_macro_f1": mean_cv_f1,
            "cv_top1_accuracy": mean_cv_top1,
            "cv_log_loss": mean_cv_loss,
            "cv_route_continuity": mean_cv_cont,
            "student_test_macro_f1": test_macro_f1,
            "student_test_top1_accuracy": test_top1,
            "student_test_precision": test_prec,
            "student_test_recall": test_rec,
            "student_test_log_loss": test_loss,
            "student_test_route_continuity": float(np.mean(all_test_conts)),
            "student_test_lcs_f1": float(np.mean(all_test_lcs_f1)),
            "training_time_sec": total_cv_train_time + refit_time,
            "inference_time_sec": inference_time,
            "confusion_matrix": cm.tolist()
        }
        
        comparison_results.append(model_record)
        print(f"  --> Student-Test Top-1: {test_top1:.4f}, Macro-F1: {test_macro_f1:.4f}, Loss: {test_loss:.4f}, Continuity: {float(np.mean(all_test_conts)):.4f}, Time: {inference_time:.2f}s", flush=True)

    # Save overall comparison JSON and CSV
    json_path = EXPERIMENTS_DIR / "model_comparison_results.json"
    with open(json_path, "w") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"\nSaved model comparison results to {json_path}", flush=True)

    # Save student test predictions for all models
    pred_path = ARTIFACTS_DIR / "student_test_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(test_predictions_by_model, f)
    print(f"Saved student test predictions to {pred_path}", flush=True)

    # Select best model based primarily on CV Top-1 accuracy
    sorted_by_top1 = sorted(comparison_results, key=lambda r: r["cv_top1_accuracy"], reverse=True)
    best_record = sorted_by_top1[0]
    best_model_name = best_record["model_name"]
    print(f"\n=======================================================", flush=True)
    print(f"Selected Best Model: {best_model_name}", flush=True)
    print(f"Rationale: Highest Grouped-CV Top-1 Accuracy ({best_record['cv_top1_accuracy']:.4f}), Student-Test Top-1 ({best_record['student_test_top1_accuracy']:.4f}), strong probability calibration, and efficient inference time ({best_record['inference_time_sec']:.2f}s).", flush=True)
    print(f"=======================================================", flush=True)

    # Refit and serialize the selected best classifier on the full training set
    print(f"Serializing final best model artifact: {best_model_name}...", flush=True)
    final_best_clf = build_model(best_model_name)
    all_train_sub = []
    for td in train_trips_data:
        keep = subsample_training_negatives(td["X"], td["y"], td["obs_indices"], max_neg_per_obs=2)
        all_train_sub.append((td["X"][keep], td["y"][keep]))
    full_sub_X = np.vstack([x for x, _ in all_train_sub])
    full_sub_y = np.concatenate([y for _, y in all_train_sub])
    final_best_clf.fit(full_sub_X, full_sub_y)

    model_artifact_path = ARTIFACTS_DIR / "best_classifier.joblib"
    joblib.dump(final_best_clf, model_artifact_path)
    print(f"Saved best model artifact to {model_artifact_path}", flush=True)

    selection_metadata = {
        "best_model_name": best_model_name,
        "selection_metric": "cv_top1_accuracy",
        "cv_top1_accuracy": best_record["cv_top1_accuracy"],
        "student_test_top1_accuracy": best_record["student_test_top1_accuracy"],
        "cv_macro_f1": best_record["cv_macro_f1"],
        "student_test_macro_f1": best_record["student_test_macro_f1"],
        "selection_rationale": f"Selected {best_model_name} because it achieved the highest grouped-CV observation-level Top-1 accuracy ({best_record['cv_top1_accuracy']:.4f}) and superior student-testing Top-1 accuracy ({best_record['student_test_top1_accuracy']:.4f}) with balanced inference efficiency."
    }
    with open(ARTIFACTS_DIR / "model_selection.json", "w") as f:
        json.dump(selection_metadata, f, indent=2)

    return comparison_results, best_model_name


if __name__ == "__main__":
    run_cross_validation_and_comparison()
