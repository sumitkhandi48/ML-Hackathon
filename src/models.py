"""
Model Definitions for PathSnap Trace-Lite
Implements exactly five distinct algorithms from different families:
1. Gaussian Naive Bayes (Probabilistic)
2. Logistic Regression (Linear)
3. Decision Tree (Tree-based)
4. Extra Trees Classifier (Ensemble)
5. HistGradientBoostingClassifier (Boosting)
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

MODEL_CONFIGS = {
    "Gaussian Naive Bayes": {
        "family": "Probabilistic",
        "description": "Gaussian Naive Bayes classifier with standard scaling on continuous geospatial features.",
        "params": {
            "var_smoothing": 1e-8
        }
    },
    "Logistic Regression": {
        "family": "Linear",
        "description": "L2-regularized Logistic Regression with L-BFGS solver and standard feature scaling.",
        "params": {
            "C": 1.0,
            "max_iter": 500,
            "solver": "lbfgs",
            "random_state": 42
        }
    },
    "Decision Tree": {
        "family": "Tree-based",
        "description": "Single decision tree with bounded depth to capture non-linear candidate selection thresholds.",
        "params": {
            "max_depth": 10,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "random_state": 42
        }
    },
    "Extra Trees": {
        "family": "Ensemble",
        "description": "Extremely Randomized Trees ensemble for variance reduction and robust decision boundaries.",
        "params": {
            "n_estimators": 50,
            "max_depth": 12,
            "min_samples_split": 15,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": 42
        }
    },
    "HistGradientBoosting": {
        "family": "Boosting",
        "description": "Histogram-based Gradient Boosting for fast, memory-efficient non-linear sequential error correction.",
        "params": {
            "max_iter": 80,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "learning_rate": 0.1,
            "random_state": 42
        }
    }
}


def build_model(model_name: str, custom_params: dict = None):
    """
    Instantiate model pipeline based on algorithm name and hyperparameters.
    """
    config = MODEL_CONFIGS[model_name]
    params = config["params"].copy()
    if custom_params:
        params.update(custom_params)

    if model_name == "Gaussian Naive Bayes":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB(**params))
        ])
    elif model_name == "Logistic Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params))
        ])
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(**params)
    elif model_name == "Extra Trees":
        return ExtraTreesClassifier(**params)
    elif model_name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
