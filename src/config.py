"""
Configuration for PathSnap Trace-Lite
PES University PathSnap Hackathon
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(r"E:\ML Hackathon")
STUDENT_DATA_DIR = BASE_DIR / "Student folder" / "Student folder"
TRAIN_DIR = STUDENT_DATA_DIR / "train"
TEST_DIR = STUDENT_DATA_DIR / "student_test"

# Evaluator Test Set
EVALUATOR_TEST_DIR = BASE_DIR / "evaluator_test" / "evaluator_test"
if not EVALUATOR_TEST_DIR.exists():
    EVALUATOR_TEST_DIR = BASE_DIR / "evaluator_test"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
EDA_DIR = BASE_DIR / "eda"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
MAPS_DIR = BASE_DIR / "maps"
REPORT_DIR = BASE_DIR / "report"

# Ensure output directories exist
for p in [ARTIFACTS_DIR, EDA_DIR, EXPERIMENTS_DIR, MAPS_DIR, REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Coordinate projection
EARTH_RADIUS_METERS = 6371000.0

# Candidate generation parameters
INITIAL_SEARCH_RADIUS = 50.0  # meters
FALLBACK_SEARCH_RADIUS = 150.0  # meters
TOP_K_CANDIDATES = 8  # bounded candidate count per observation

# Cross-validation
CV_FOLDS = 3
RANDOM_SEED = 42

# Models to evaluate
MODEL_NAMES = [
    "Gaussian Naive Bayes",
    "Logistic Regression",
    "Decision Tree",
    "Extra Trees",
    "HistGradientBoosting"
]
