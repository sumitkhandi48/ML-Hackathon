"""
FastAPI Backend Application for PathSnap Trace-Lite
Provides endpoints: GET /health, GET /trips, POST /predict.
Supports deployment across LAN IP for multi-laptop evaluation.
"""
import time
import os
import json
from pathlib import Path
from typing import Optional
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import ARTIFACTS_DIR, STUDENT_DATA_DIR, TEST_DIR
from src.data_loader import load_trip, list_trips
from src.candidate_gen import generate_trip_candidates
from src.features import extract_candidate_features
from src.viterbi import ViterbiRouteDecoder
from backend.evaluator_service import EvaluatorService

app = FastAPI(title="PathSnap Trace-Lite API", version="1.0.0")

# Enable CORS for multi-device/LAN IP communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and evaluator instances
model = None
evaluator = EvaluatorService()

def get_model():
    global model
    if model is None:
        tuned_path = ARTIFACTS_DIR / "final_tuned_model.joblib"
        best_path = ARTIFACTS_DIR / "best_classifier.joblib"
        target_path = tuned_path if tuned_path.exists() else best_path
        if target_path.exists():
            print(f"[BACKEND INIT] Loading model from {target_path}...", flush=True)
            model = joblib.load(target_path)
        else:
            print("[BACKEND WARNING] Model artifact not found on disk yet.", flush=True)
    return model


class PredictRequest(BaseModel):
    trip_id: str
    partition: Optional[str] = "student_test"
    use_hmm: Optional[bool] = True


@app.get("/health")
def health():
    """Health check endpoint."""
    m = get_model()
    return {
        "status": "ok",
        "service": "PathSnap Trace-Lite API",
        "model_loaded": m is not None,
        "device": "CPU",
        "timestamp": time.time()
    }


@app.get("/trips")
def get_available_trips(partition: str = "student_test"):
    """
    Dynamically discover trip IDs in the target partition.
    Does not hardcode or access unsupplied data.
    """
    p = STUDENT_DATA_DIR / partition
    if not p.exists():
        p = TEST_DIR
    trips = sorted([d.name for d in p.iterdir() if d.is_dir()])
    return {
        "partition": partition,
        "count": len(trips),
        "trips": trips
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Perform map-matching inference on a trip.
    Generates candidates, extracts features, predicts arc probabilities,
    and reconstructs sequential route with HMM / Viterbi.
    Only after inference completes, invokes the separate EvaluatorService for metrics.
    """
    t_start = time.time()
    print(f"\n[INFERENCE REQUEST] Received request for Trip ID: {req.trip_id} (Partition: {req.partition})", flush=True)

    clf = get_model()
    if clf is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded on server.")

    # 1. Load genuine trip track and graph
    try:
        trip = load_trip(req.trip_id, req.partition)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to load trip {req.trip_id}: {str(e)}")

    T = len(trip.track_times)
    print(f"  [DATA LOADED] Points: {T:,} GPS observations, Nodes: {trip.num_nodes:,}, Arcs: {trip.num_arcs:,}", flush=True)

    # Check for time gaps > 5.0 seconds
    warnings = []
    if T > 1:
        dts = np.diff(trip.track_times)
        gap_indices = np.where(dts > 5.0)[0]
        for g_idx in gap_indices:
            warnings.append({
                "obs_index": int(g_idx),
                "timestamp": float(trip.track_times[g_idx]),
                "gap_seconds": float(dts[g_idx]),
                "message": f"Time gap of {dts[g_idx]:.1f}s between obs {g_idx} and {g_idx+1}"
            })

    # 2. Candidate generation
    candidates = generate_trip_candidates(trip, k=8)

    # 3. Candidate feature extraction
    X, obs_indices, cand_arcs, cand_ranks = extract_candidate_features(trip, candidates)

    # 4. Model probability prediction
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[:, 1]
    else:
        scores = clf.decision_function(X)
        probs = 1.0 / (1.0 + np.exp(-scores))

    # 5. Route decoding: Classifier-only vs HMM / Viterbi
    if req.use_hmm:
        cand_probs_per_obs = []
        ptr = 0
        for t in range(T):
            num_c = len(candidates[t].arc_ids)
            if num_c > 0:
                cand_probs_per_obs.append(probs[ptr:ptr+num_c])
            else:
                cand_probs_per_obs.append(np.array([], dtype=np.float32))
            ptr += num_c

        decoder = ViterbiRouteDecoder(trip, beta=20.0, max_speed=35.0)
        predicted_arcs, reconstructed_route, viterbi_diag = decoder.decode(candidates, cand_probs_per_obs)
        pred_arc_list = predicted_arcs.tolist()
        route_arc_sequence = reconstructed_route
    else:
        predicted_arcs = np.full(T, -1, dtype=np.int32)
        ptr = 0
        for t in range(T):
            num_c = len(candidates[t].arc_ids)
            if num_c > 0:
                c_p = probs[ptr:ptr+num_c]
                c_a = cand_arcs[ptr:ptr+num_c]
                predicted_arcs[t] = c_a[np.argmax(c_p)]
            ptr += num_c
        pred_arc_list = predicted_arcs.tolist()
        from src.metrics import compress_sequence
        route_arc_sequence = compress_sequence(pred_arc_list)

    inference_time_ms = round((time.time() - t_start) * 1000.0, 1)
    print(f"  [INFERENCE COMPLETE] Decoded in {inference_time_ms} ms. Predicted {len(route_arc_sequence)} unique/reconstructed arcs.", flush=True)

    # 6. Build geo-coordinates for Leaflet map display
    # Raw GPS coordinates: [[lat, lon], ...]
    gps_coords = [[float(trip.track_lats[t]), float(trip.track_lons[t])] for t in range(T)]

    # Predicted route segments: list of [[lat1, lon1], [lat2, lon2]]
    # Unproject endpoints of predicted arcs
    pred_segments = []
    for arc_id in route_arc_sequence:
        if 0 <= arc_id < trip.num_arcs:
            u = trip.arcs_from[arc_id]
            v = trip.arcs_to[arc_id]
            lat1 = float(trip.nodes_lats[u])
            lon1 = float(trip.nodes_lons[u])
            lat2 = float(trip.nodes_lats[v])
            lon2 = float(trip.nodes_lons[v])
            pred_segments.append([[lat1, lon1], [lat2, lon2]])

    # Nearby road network segments for visual context (subsampled around bounding box)
    sub_lines = []
    min_x, max_x = np.min(trip.track_x) - 300, np.max(trip.track_x) + 300
    min_y, max_y = np.min(trip.track_y) - 300, np.max(trip.track_y) + 300
    nearby = (
        (trip.x1 >= min_x) & (trip.x1 <= max_x) &
        (trip.y1 >= min_y) & (trip.y1 <= max_y)
    )
    n_idx = np.where(nearby)[0]
    step = max(1, len(n_idx) // 400)
    for idx in n_idx[::step]:
        u = trip.arcs_from[idx]
        v = trip.arcs_to[idx]
        sub_lines.append([[float(trip.nodes_lats[u]), float(trip.nodes_lons[u])],
                          [float(trip.nodes_lats[v]), float(trip.nodes_lons[v])]])

    # 7. Isolated Server-Side Evaluation (CALLED ONLY AFTER INFERENCE IS COMPLETE)
    # Strictly isolated: Never returns reference route or per-point labels!
    eval_metrics = evaluator.evaluate_trip(req.trip_id, pred_arc_list, req.partition)
    print(f"  [EVALUATION METRICS] Top-1: {eval_metrics.get('top1_accuracy_percent', 'N/A')}%, Continuity: {eval_metrics.get('route_continuity_percent', 'N/A')}%, LCS F1: {eval_metrics.get('sequence_lcs_f1_percent', 'N/A')}%", flush=True)

    return {
        "trip_id": req.trip_id,
        "num_observations": T,
        "inference_time_ms": inference_time_ms,
        "gps_points": gps_coords,
        "predicted_route_segments": pred_segments,
        "road_network_segments": sub_lines,
        "metrics": eval_metrics,
        "warnings": warnings,
        "status": "success"
    }


# Mount static frontend
frontend_dir = Path(r"E:\ML Hackathon\frontend")
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
