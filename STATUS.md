# PathSnap Trace-Lite — Implementation Status

**Project:** PES University PathSnap Hackathon  
**Target Completion:** November 18, 2026, 1:30 PM IST  
**Last Updated:** November 18, 2026, 12:15 PM IST  

---

## 1. Overall Progress Summary

| Stage | Description | Status | Evidence / Artifacts |
|---|---|---|---|
| **Stage 1** | Data Parsing, Spatial Indexing, Nearest Baseline, EDA, Evaluation Metrics | IN PROGRESS | Starting pipeline, EDA scripts, baseline evaluator |
| **Stage 2** | Reference Alignment, Candidate Features, 5 Models (GNB, LR, DT, ET, HGB) | PENDING | Awaiting Stage 1 feature caching |
| **Stage 3A** | Bayesian Hyperparameter Optimization (Optuna TPE) | PENDING | Awaiting Stage 2 model selection |
| **Stage 3B** | Custom Log-Space Viterbi / HMM Route Decoder | PENDING | Parallel development with Stage 3A |
| **Stage 4** | FastAPI Backend (`/health`, `/trips`, `/predict`) + Leaflet UI | PENDING | Implementation ready after model serialization |
| **Stage 5** | Folium Maps, Evaluation Graphs, Consolidated PDF Technical Report | PENDING | Output generators prepared |
| **Stage 6** | End-to-End Verification, Git Checkpoints, Repository Readme & Viva Notes | PENDING | Pre-deadline verification |

---

## 2. Completed Milestones & Execution Evidence

- Extracted and reviewed all official guidelines (`Main Guidelines`, `Dataset Exploration`, `Deliverables`, `Models and Training Guidelines`, `Assessment and Plagiarism Policy`).
- Verified workspace paths and data directory: 80 training trips, 10 student-testing trips in `Student folder/Student folder/`.
- Verified required libraries: `numpy`, `scipy`, `shapely`, `sklearn`, `optuna`, `fastapi`, `uvicorn`, `folium`, `matplotlib`, `seaborn`, `reportlab`, `PIL`.
- Configured repository `.gitignore` and initialized project tracking.

---

## 3. Active Blockers & Mitigations

- *None at present.* All required dependencies and data files are accessible locally.

---

## 4. Immediate Next Steps

1. Implement `src/geo_utils.py` and `src/data_loader.py` for local metric projection and compact road network graph indexing with Shapely STRtree.
2. Implement Stage 1 EDA suite to generate missing values, GPS observation counts, graph connectivity, timestamp gaps, and Folium trajectory plots.
3. Implement candidate generation (point-to-segment distance, 6–10 candidates) and nearest-road baseline evaluation.
