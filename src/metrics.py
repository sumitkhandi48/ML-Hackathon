"""
Evaluation Metrics for PathSnap Trace-Lite
Exact mathematical definitions for observation-level and candidate-level metrics,
route continuity, and sequence-level LCS metrics.
"""
from typing import Sequence
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss

def compute_lcs_length(seq1: Sequence[int], seq2: Sequence[int]) -> int:
    """Compute length of Longest Common Subsequence between two integer sequences."""
    n = len(seq1)
    m = len(seq2)
    if n == 0 or m == 0:
        return 0
    # Use two rows DP for memory efficiency
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        s1_val = seq1[i - 1]
        for j in range(1, m + 1):
            if s1_val == seq2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (m + 1)
    return prev[m]


def compress_sequence(seq: Sequence[int]) -> list[int]:
    """Compress consecutive repeated elements: [1, 1, 2, 2, 3] -> [1, 2, 3]. Filter out -1."""
    compressed = []
    for x in seq:
        if x != -1 and (not compressed or compressed[-1] != x):
            compressed.append(int(x))
    return compressed


def evaluate_predictions(
    y_true_obs: np.ndarray,          # True reference arc per observation (T,)
    y_pred_obs: np.ndarray,          # Predicted arc per observation (T,)
    cand_y_true: np.ndarray,         # Binary ground-truth per candidate (N_cand,)
    cand_y_prob: np.ndarray,         # Predicted probability per candidate (N_cand,)
    cand_coverage_mask: np.ndarray,  # Boolean whether true arc was in candidates (T,)
    arcs_from: np.ndarray,           # Arc source nodes (M,)
    arcs_to: np.ndarray,             # Arc target nodes (M,)
    reference_route: np.ndarray = None # Ground truth sequence (L,)
) -> dict:
    """
    Comprehensive evaluation matching official requirements.
    """
    T = len(y_true_obs)
    assert len(y_pred_obs) == T, "Observation array length mismatch"
    
    # 1. Observation-level Top-1 Accuracy
    # Denominator: total observations T
    correct_obs = (y_pred_obs == y_true_obs) & (y_true_obs != -1) & (y_pred_obs != -1)
    top1_accuracy = float(np.sum(correct_obs) / T) if T > 0 else 0.0
    
    # 2. Candidate Miss Rate
    # Denominator: total observations T
    miss_rate = float(1.0 - np.mean(cand_coverage_mask)) if T > 0 else 0.0
    
    # 3. Candidate-level Precision, Recall, Macro F1, Log Loss
    # Threshold at 0.5 for binary classification
    cand_y_pred_binary = (cand_y_prob >= 0.5).astype(int)
    
    # If all predicted 0, ensure at least rank 1 per obs if needed, but standard metrics use binary
    prec = float(precision_score(cand_y_true, cand_y_pred_binary, zero_division=0))
    rec = float(recall_score(cand_y_true, cand_y_pred_binary, zero_division=0))
    f1_macro = float(f1_score(cand_y_true, cand_y_pred_binary, average="macro", zero_division=0))
    
    # Clamped log loss to avoid inf
    eps = 1e-7
    clamped_prob = np.clip(cand_y_prob, eps, 1.0 - eps)
    try:
        loss = float(log_loss(cand_y_true, clamped_prob))
    except Exception:
        loss = 0.0
        
    # 4. Route Continuity
    # Check transitions between consecutive observations
    if T <= 1:
        continuity = 1.0
    else:
        continuous_steps = 0
        total_transitions = T - 1
        for t in range(total_transitions):
            a1 = y_pred_obs[t]
            a2 = y_pred_obs[t + 1]
            if a1 == -1 or a2 == -1:
                continue
            if a1 == a2:
                continuous_steps += 1
            else:
                to_node = arcs_to[a1]
                from_node = arcs_from[a2]
                if to_node == from_node:
                    continuous_steps += 1
        continuity = float(continuous_steps / total_transitions)
        
    # 5. Sequence-level LCS F1
    pred_seq = compress_sequence(y_pred_obs)
    if reference_route is not None:
        target_seq = compress_sequence(reference_route)
    else:
        target_seq = compress_sequence(y_true_obs)
        
    lcs_len = compute_lcs_length(pred_seq, target_seq)
    lcs_prec = float(lcs_len / len(pred_seq)) if len(pred_seq) > 0 else 0.0
    lcs_rec = float(lcs_len / len(target_seq)) if len(target_seq) > 0 else 0.0
    lcs_f1 = float(2.0 * lcs_prec * lcs_rec / (lcs_prec + lcs_rec)) if (lcs_prec + lcs_rec) > 0 else 0.0
    
    return {
        "top1_accuracy": top1_accuracy,
        "candidate_macro_f1": f1_macro,
        "candidate_precision": prec,
        "candidate_recall": rec,
        "log_loss": loss,
        "candidate_miss_rate": miss_rate,
        "route_continuity": continuity,
        "lcs_f1": lcs_f1,
        "lcs_precision": lcs_prec,
        "lcs_recall": lcs_rec,
        "pred_sequence_length": len(pred_seq),
        "target_sequence_length": len(target_seq),
        "lcs_matched_arcs": lcs_len
    }
