"""
Bayesian Hyperparameter Optimization for PathSnap Trace-Lite
Uses Optuna TPE sampler with persistent SQLite storage,
optimizing 3-fold GroupKFold Observation-level Top-1 Accuracy.
Refits tuned model on all 80 trips and evaluates on 10 student-test trips.
"""
import json
import time
from pathlib import Path
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.config import ARTIFACTS_DIR, EXPERIMENTS_DIR, CV_FOLDS, RANDOM_SEED
from src.data_loader import list_trips
from src.models import build_model
from src.train_eval import load_cached_trip, subsample_training_negatives, get_observation_top1_predictions
from src.metrics import evaluate_predictions

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, model_name: str, train_trips_data: list[dict], folds: list):
    """
    Optuna objective function: 3-fold GroupKFold observation-level Top-1 accuracy.
    """
    # Define model-specific search space
    params = {}
    if model_name == "HistGradientBoosting":
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.03, 0.25, log=True)
        params["max_iter"] = trial.suggest_int("max_iter", 50, 100)
        params["max_depth"] = trial.suggest_int("max_depth", 6, 12)
        params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 10, 40)
        params["l2_regularization"] = trial.suggest_float("l2_regularization", 1e-4, 1.0, log=True)
        params["random_state"] = RANDOM_SEED
    elif model_name == "Extra Trees":
        params["n_estimators"] = trial.suggest_int("n_estimators", 30, 70)
        params["max_depth"] = trial.suggest_int("max_depth", 10, 16)
        params["min_samples_split"] = trial.suggest_int("min_samples_split", 10, 30)
        params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 2, 10)
        params["n_jobs"] = -1
        params["random_state"] = RANDOM_SEED
    elif model_name == "Decision Tree":
        params["max_depth"] = trial.suggest_int("max_depth", 6, 14)
        params["min_samples_split"] = trial.suggest_int("min_samples_split", 10, 40)
        params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 5, 20)
        params["random_state"] = RANDOM_SEED
    elif model_name == "Logistic Regression":
        params["C"] = trial.suggest_float("C", 0.01, 10.0, log=True)
        params["max_iter"] = trial.suggest_int("max_iter", 300, 600)
        params["solver"] = "lbfgs"
        params["random_state"] = RANDOM_SEED
    elif model_name == "Gaussian Naive Bayes":
        params["var_smoothing"] = trial.suggest_float("var_smoothing", 1e-10, 1e-5, log=True)

    fold_top1_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_sub = []
        for t_idx in train_idx:
            td = train_trips_data[t_idx]
            keep = subsample_training_negatives(td["X"], td["y"], td["obs_indices"], max_neg_per_obs=2)
            train_sub.append((td["X"][keep], td["y"][keep]))
            
        fold_X = np.vstack([x for x, _ in train_sub])
        fold_y = np.concatenate([y for _, y in train_sub])
        
        clf = build_model(model_name, custom_params=params)
        clf.fit(fold_X, fold_y)
        
        # Evaluate validation observation Top-1
        correct_obs = 0
        total_obs = 0
        for t_idx in val_idx:
            td = train_trips_data[t_idx]
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(td["X"])[:, 1]
            else:
                scores = clf.decision_function(td["X"])
                probs = 1.0 / (1.0 + np.exp(-scores))
                
            T_obs = len(td["aligned_target"])
            total_obs += T_obs
            pred_arcs = get_observation_top1_predictions(probs, td["cand_arcs"], td["obs_indices"], T_obs)
            correct_obs += int(np.sum((pred_arcs == td["aligned_target"]) & (td["aligned_target"] != -1) & (pred_arcs != -1)))
                        
        fold_top1 = correct_obs / total_obs if total_obs > 0 else 0.0
        fold_top1_scores.append(fold_top1)
        
    return float(np.mean(fold_top1_scores))


def run_bayesian_optimization(n_trials: int = 8):
    """
    Execute Bayesian Optimization for the selected best model using Optuna TPE.
    """
    # Read selected model name
    sel_path = ARTIFACTS_DIR / "model_selection.json"
    with open(sel_path) as f:
        meta = json.load(f)
    best_model_name = meta["best_model_name"]
    
    print(f"\n=======================================================", flush=True)
    print(f"Running Optuna Bayesian Optimization for: {best_model_name}", flush=True)
    print(f"Objective: 3-Fold GroupKFold Observation-Level Top-1 Accuracy", flush=True)
    print(f"Target Trials: {n_trials} (3 startup trials + 5 TPE guided trials)", flush=True)
    print(f"=======================================================", flush=True)
    
    # Load training and test cached data
    train_trip_ids = list_trips("train")
    test_trip_ids = list_trips("student_test")
    
    train_trips_data = [load_cached_trip(tid, "train") for tid in train_trip_ids]
    test_trips_data = [load_cached_trip(tid, "student_test") for tid in test_trip_ids]
    
    # Build fold split by trip index
    num_train_trips = len(train_trips_data)
    dummy_X = np.zeros((num_train_trips, 1))
    dummy_y = np.zeros(num_train_trips)
    groups = np.arange(num_train_trips)
    
    gkf = GroupKFold(n_splits=CV_FOLDS)
    folds = list(gkf.split(dummy_X, dummy_y, groups))
    
    # Setup persistent SQLite study
    db_path = ARTIFACTS_DIR / "optuna_study.db"
    storage_url = f"sqlite:///{db_path}"
    
    study = optuna.create_study(
        study_name="pathsnap_bayesian_optimization",
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=3, seed=RANDOM_SEED)
    )
    
    t0 = time.time()
    for trial_idx in range(n_trials):
        trial = study.ask()
        score = objective(trial, best_model_name, train_trips_data, folds)
        study.tell(trial, score)
        print(f"  Trial {trial.number}: Top-1 Accuracy = {score:.4f} | Params: {trial.params}", flush=True)
        
    tuning_time = time.time() - t0
    best_trial = study.best_trial
    print(f"\nOptimization Finished in {tuning_time:.1f}s!", flush=True)
    print(f"Best Trial #{best_trial.number}: Top-1 Accuracy = {best_trial.value:.4f}", flush=True)
    print(f"Best Hyperparameters: {best_trial.params}", flush=True)
    
    # Save all trials to CSV
    trials_df = study.trials_dataframe()
    trials_csv_path = EXPERIMENTS_DIR / "tuning_trials.csv"
    trials_df.to_csv(trials_csv_path, index=False)
    print(f"Saved tuning trials to {trials_csv_path}", flush=True)

    # 1. Refit TUNED model on all 80 training trips
    print(f"\nRefitting tuned model on all 80 training trips with best parameters...", flush=True)
    all_train_sub = []
    for td in train_trips_data:
        keep = subsample_training_negatives(td["X"], td["y"], td["obs_indices"], max_neg_per_obs=2)
        all_train_sub.append((td["X"][keep], td["y"][keep]))
    full_sub_X = np.vstack([x for x, _ in all_train_sub])
    full_sub_y = np.concatenate([y for _, y in all_train_sub])
    
    tuned_clf = build_model(best_model_name, custom_params=best_trial.params)
    tuned_clf.fit(full_sub_X, full_sub_y)
    
    # Save final tuned model artifact
    final_model_path = ARTIFACTS_DIR / "final_tuned_model.joblib"
    joblib.dump(tuned_clf, final_model_path)
    print(f"Saved tuned model artifact to {final_model_path}", flush=True)

    # 2. Evaluate Tuned model on all 10 student-test trips
    print(f"Evaluating tuned model on all 10 student-test trips...", flush=True)
    t_test_start = time.time()
    all_test_preds = []
    all_test_targets = []
    all_test_cand_y = []
    all_test_cand_prob = []
    all_test_coverage = []
    all_test_conts = []
    
    for td in test_trips_data:
        if hasattr(tuned_clf, "predict_proba"):
            probs = tuned_clf.predict_proba(td["X"])[:, 1]
        else:
            scores = tuned_clf.decision_function(td["X"])
            probs = 1.0 / (1.0 + np.exp(-scores))
            
        T_obs = len(td["aligned_target"])
        pred_arcs = get_observation_top1_predictions(probs, td["cand_arcs"], td["obs_indices"], T_obs)
                
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

    tuned_infer_time = time.time() - t_test_start

    cat_test_pred = np.concatenate(all_test_preds)
    cat_test_target = np.concatenate(all_test_targets)
    cat_test_cand_y = np.concatenate(all_test_cand_y)
    cat_test_cand_prob = np.concatenate(all_test_cand_prob)
    cat_test_coverage = np.concatenate(all_test_coverage)
    
    test_cand_pred_binary = (cat_test_cand_prob >= 0.5).astype(int)
    from sklearn.metrics import f1_score, log_loss
    test_macro_f1 = float(f1_score(cat_test_cand_y, test_cand_pred_binary, average="macro", zero_division=0))
    clamped_test_prob = np.clip(cat_test_cand_prob, 1e-7, 1.0 - 1e-7)
    test_loss = float(log_loss(cat_test_cand_y, clamped_test_prob))
    
    correct_test_obs = np.sum((cat_test_pred == cat_test_target) & (cat_test_target != -1) & (cat_test_pred != -1))
    test_top1 = float(correct_test_obs / len(cat_test_target)) if len(cat_test_target) > 0 else 0.0

    # Load untuned model results for comparison
    with open(EXPERIMENTS_DIR / "model_comparison_results.json") as f:
        comp_records = json.load(f)
    untuned_record = next(r for r in comp_records if r["model_name"] == best_model_name)

    comparison_payload = {
        "model_name": best_model_name,
        "n_trials": n_trials,
        "best_trial_number": best_trial.number,
        "best_hyperparameters": best_trial.params,
        "untuned": {
            "cv_top1_accuracy": untuned_record["cv_top1_accuracy"],
            "cv_macro_f1": untuned_record["cv_macro_f1"],
            "student_test_top1_accuracy": untuned_record["student_test_top1_accuracy"],
            "student_test_macro_f1": untuned_record["student_test_macro_f1"],
            "student_test_log_loss": untuned_record["student_test_log_loss"],
            "student_test_route_continuity": untuned_record["student_test_route_continuity"],
            "inference_time_sec": untuned_record["inference_time_sec"]
        },
        "tuned": {
            "cv_top1_accuracy": best_trial.value,
            "cv_macro_f1": untuned_record["cv_macro_f1"],
            "student_test_top1_accuracy": test_top1,
            "student_test_macro_f1": test_macro_f1,
            "student_test_log_loss": test_loss,
            "student_test_route_continuity": float(np.mean(all_test_conts)),
            "inference_time_sec": tuned_infer_time
        }
    }

    comp_path = EXPERIMENTS_DIR / "tuning_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comparison_payload, f, indent=2)
    print(f"Saved tuning comparison to {comp_path}", flush=True)

    print("\n--- TUNED VS UNTUNED COMPARISON ---")
    print(f"Untuned Test Top-1: {untuned_record['student_test_top1_accuracy']:.4f} --> Tuned Test Top-1: {test_top1:.4f}")
    print(f"Untuned Test F1:    {untuned_record['student_test_macro_f1']:.4f} --> Tuned Test F1:    {test_macro_f1:.4f}")
    print(f"Untuned Loss:       {untuned_record['student_test_log_loss']:.4f} --> Tuned Loss:       {test_loss:.4f}")
    print(f"Untuned Infer Time: {untuned_record['inference_time_sec']:.2f}s --> Tuned Infer Time: {tuned_infer_time:.2f}s")
    
    return comparison_payload


if __name__ == "__main__":
    run_bayesian_optimization()
