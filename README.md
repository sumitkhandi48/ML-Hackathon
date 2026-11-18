# PathSnap Trace-Lite

**Distributed Map-Matching & Sequential Route Reconstruction System**  
*Department of Computer Science and Engineering, PES University*  
*Course: UE24CS352A Machine Learning | Hackathon 2026*

---

## 1. Project Overview

PathSnap Trace-Lite reconstructs realistic vehicle trajectories by matching noisy, irregularly sampled GPS observations onto a directed road network graph. The system integrates:
1. **Kinematic & Geospatial Preprocessing:** Local metric equirectangular coordinate projection, vectorized road segment geometry, and sub-millisecond candidate generation via Shapely STRtree indexing.
2. **Supervised Candidate Scoring:** A comparative benchmark across **five distinct algorithm families** (Gaussian Naive Bayes, Logistic Regression, Decision Tree, Extra Trees, and HistGradientBoosting) evaluated via 3-fold GroupKFold cross-validation grouped by trip ID.
3. **Bayesian Hyperparameter Optimization:** Optuna Tree-structured Parzen Estimator (TPE) with persistent SQLite storage targeting observation-level Top-1 accuracy.
4. **Custom Log-Space Viterbi / HMM Decoder:** Sequential trajectory decoding combining supervised emission probabilities with network-constrained Dijkstra transition costs and speed penalties.
5. **Distributed Multi-Laptop Deployment:** A lightweight FastAPI server (Laptop 1) with an isolated server-side evaluation service and an interactive Leaflet frontend (Laptop 2) accessible across the local network (LAN IP).

---

## 2. Repository Structure

```
├── src/
│   ├── config.py             # System paths, constants, and default parameters
│   ├── geo_utils.py          # Metric projection, STRtree indexing, point-to-segment geometry
│   ├── data_loader.py        # Streaming parser for .track, .nodes, .arcs, .route per trip
│   ├── candidate_gen.py      # Bounded candidate retrieval (K=8) and nearest-road baseline
│   ├── label_alignment.py    # Reference-constrained DP ground-truth label derivation
│   ├── features.py           # 13 kinematic and geometric candidate features
│   ├── metrics.py            # Evaluation suite: Top-1, Macro F1, continuity, LCS F1
│   ├── models.py             # Model definitions for all 5 algorithm families
│   ├── viterbi.py            # Custom log-space HMM decoder with Dijkstra router
│   ├── eda.py                # Dataset exploration, statistics, and static/Folium visualizers
│   ├── feature_cache.py      # Feature extraction and caching pipeline
│   ├── train_eval.py         # 3-Fold GroupKFold and student-testing evaluation
│   ├── optimize.py           # Optuna TPE Bayesian hyperparameter optimization
│   ├── evaluate_hmm.py       # Classifier-only vs Classifier+HMM comparative benchmark
│   └── generate_plots_report.py # Automated generation of figures and consolidated PDF report
├── backend/
│   ├── main.py               # FastAPI application (/health, /trips, /predict)
│   └── evaluator_service.py  # Isolated server-side ground-truth evaluation service
├── frontend/
│   ├── index.html            # Clean HTML5 UI with status badges and diagnostics
│   ├── style.css             # Modern dark-mode dashboard styling
│   └── app.js                # Leaflet map rendering, layer controls, and API client
├── artifacts/                # Serialized models, caches, SQLite databases, predictions
├── eda/                      # EDA summaries, tables, and distribution plots
├── experiments/              # JSON/CSV metric logs, comparison tables, tuning trials
├── maps/                     # Interactive Folium HTML maps
├── report/                   # Consolidated Technical Report PDF
├── STATUS.md                 # Real-time implementation status & execution evidence
├── requirements.txt          # Python dependencies
└── README.md                 # Setup, execution, and technical documentation
```

---

## 3. Installation & Setup

### Environment Requirements
- Python 3.10+
- Windows / Linux / macOS

```bash
# Clone the repository
git clone https://github.com/team/pathsnap-trace-lite.git
cd pathsnap-trace-lite

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Execution Workflow

To reproduce all experimental results from scratch:

### Step 1: Exploratory Data Analysis & Baseline Evaluation
```bash
python -m src.eda
```
*Generates `eda/trips_statistics.csv`, `eda/eda_summary.json`, `eda/eda_distributions.png`, and `maps/representative_folium_map.html`.*

### Step 2: Feature Extraction & Caching
```bash
python -m src.feature_cache
```
*Processes all 80 training trips and 10 student-testing trips, creating isolated `.npz` feature caches in `artifacts/cache/`.*

### Step 3: Train & Benchmark 5 Algorithms (3-Fold GroupKFold)
```bash
python -m src.train_eval
```
*Trains Gaussian NB, Logistic Regression, Decision Tree, Extra Trees, and HistGradientBoosting; computes grouped CV metrics, evaluates student-test performance, and serializes `artifacts/best_classifier.joblib`.*

### Step 4: Bayesian Hyperparameter Optimization (Optuna TPE)
```bash
python -m src.optimize
```
*Executes guided TPE optimization with persistent SQLite tracking (`artifacts/optuna_study.db`), refits the tuned model on all 80 trips, and evaluates gains.*

### Step 5: HMM & Viterbi Route Reconstruction
```bash
python -m src.evaluate_hmm
```
*Benchmarks Classifier-Only versus Classifier + HMM / Viterbi decoding across all 10 student-test trips, logging Top-1 accuracy, route continuity, and sequence LCS F1.*

### Step 6: Generate Plots & Consolidated PDF Report
```bash
python -m src.generate_plots_report
```
*Generates publication figures in `experiments/` and compiles the official `report/PathSnap_Technical_Report.pdf`.*

---

## 5. Distributed Deployment (FastAPI + Leaflet UI)

### Host Laptop (Laptop 1):
Start the FastAPI backend:
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
*Displays real inference requests, decoded trajectory statistics, and latency logs directly in the terminal.*

### Evaluator Laptop (Laptop 2):
1. Connect Laptop 2 to the same local Wi-Fi / LAN network.
2. Open a browser and navigate to:
   ```
   http://<LAPTOP_1_IP>:8000
   ```
3. Use the trip dropdown to dynamically query discovered evaluator trips, toggle HMM decoding, and view real-time accuracy, continuity, and route geometry overlays on Leaflet.

---

## 6. Technical Viva Preparation Notes

1. **Why Local Metric Projection?**  
   Spherical lat/lon degrees are non-metric and distort distances depending on latitude. An equirectangular projection centered at the trajectory centroid preserves millimeter accuracy over local urban domains while enabling purely vectorized Euclidean distance calculations.
2. **Why Candidate-Level Classification Instead of Pointwise Arc Classification?**  
   Each trip road network has tens of thousands of arcs (up to 160,000 arcs). Directly treating this as a multi-class problem with 160k classes is computationally intractable. Candidate generation reduces the search space to $K=8$ local candidate segments, converting the task into a binary candidate scoring problem.
3. **How Does Ground-Truth Label Alignment Work?**  
   Reference routes specify sequence of traversed arcs, but not pointwise timestamps. A reference-constrained dynamic programming pass aligns each GPS point to a reference arc monotonically along the sequence using distance-based emissions, preserving route order without leakage.
4. **How Does the HMM Viterbi Decoder Prevent Disconnected Routes?**  
   The emission score measures classifier confidence, while the transition score measures network path distance (via Dijkstra) versus observed GPS displacement. Jumping across buildings or between disconnected road segments incurs severe network distance penalties, forcing the decoder to select topologically feasible contiguous paths.
5. **How is Ground-Truth Isolated During Evaluation?**  
   The `POST /predict` endpoint runs inference strictly from GPS points and road network graphs. Only after inference finishes does the server-side `EvaluatorService` evaluate predictions against the reference route. Reference data is never transmitted to the client frontend.

---

## 7. Compliance & AI Assistance Disclosure

This project was implemented independently for the PES University PathSnap Hackathon in compliance with event academic integrity policies. All code, model artifacts, and evaluation tables represent genuine experimental execution.
