import numpy as np
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Optional: XGBoost / LightGBM
_HAS_XGB = False
_HAS_LGBM = False
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    pass

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:
    pass


def classification_metrics(y_true, y_pred, y_score=None):
    """
    Works for binary + multiclass.
    - Uses macro-averaging for multiclass.
    - ROC-AUC:
        * binary: roc_auc_score(y, score_1d)
        * multiclass: roc_auc_score(y, score_2d, multi_class="ovr", average="macro")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_classes = int(len(np.unique(y_true)))
    avg = "binary" if n_classes == 2 else "macro"

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
    }

    if y_score is not None:
        try:
            if n_classes == 2:
                out["roc_auc"] = float(roc_auc_score(y_true, y_score))
            else:
                out["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
                )
        except Exception:
            pass

    return out


def _get_scores_for_auc(model, X, n_classes):
    """
    Returns scores for ROC-AUC if possible:
      - binary: shape (n_samples,)
      - multiclass: shape (n_samples, n_classes)
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if n_classes == 2:
            return proba[:, 1]
        return proba

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return scores

    return None


def run_classical_benchmarks(
    X_train, y_train, X_test, y_test,
    seed=3, cv_splits=5,
    include_xgboost=True,
    include_lightgbm=True,
    progress=True,
):
    """
    Classical baselines for binary + multiclass classification:
      - LogisticRegression (multinomial when multiclass)
      - SVC (RBF)
      - RandomForest
      - GradientBoosting
      - MLP
      - XGBoost (optional)
      - LightGBM (optional)

    Model selection metric: balanced_accuracy (works for multiclass too).
    """
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    n_classes = int(len(np.unique(y_train)))

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    models = {}

    # Logistic regression: multinomial if multiclass
    models["logreg"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=8000,
                solver="lbfgs",
                multi_class="auto",
                class_weight="balanced" if n_classes == 2 else None,
                random_state=seed
            ))
        ]),
        {"clf__C": [0.01, 0.1, 1, 10, 100]}
    )

    # SVM RBF (one-vs-one internally for multiclass)
    models["svm_rbf"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced" if n_classes == 2 else None,
                random_state=seed
            ))
        ]),
        {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1, 1]}
    )

    models["random_forest"] = (
        RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced" if n_classes == 2 else None,
            random_state=seed,
            n_jobs=-1
        ),
        {"max_depth": [None, 6, 12, 24], "min_samples_leaf": [1, 2, 5], "max_features": ["sqrt", "log2", None]}
    )

    models["grad_boost"] = (
        GradientBoostingClassifier(random_state=seed),
        {"learning_rate": [0.03, 0.1, 0.2], "n_estimators": [200, 500], "max_depth": [2, 3, 4]}
    )

    models["mlp"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                alpha=1e-4,
                max_iter=3000,
                early_stopping=True,
                random_state=seed
            ))
        ]),
        {
            "clf__hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128)],
            "clf__alpha": [1e-5, 1e-4, 1e-3],
            "clf__learning_rate_init": [1e-3, 3e-4]
        }
    )

    if include_xgboost:
        if not _HAS_XGB:
            print("[warn] xgboost not installed. Install with: pip install xgboost")
        else:
            objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
            models["xgboost"] = (
                XGBClassifier(
                    objective=objective,
                    num_class=n_classes if n_classes > 2 else None,
                    eval_metric="mlogloss" if n_classes > 2 else "logloss",
                    n_estimators=800,
                    random_state=seed,
                    n_jobs=-1,
                    tree_method="hist",
                ),
                {
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.03, 0.1],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "reg_lambda": [1.0, 5.0],
                }
            )

    if include_lightgbm:
        if not _HAS_LGBM:
            print("[warn] lightgbm not installed. Install with: pip install lightgbm")
        else:
            models["lightgbm"] = (
                LGBMClassifier(
                    objective="multiclass" if n_classes > 2 else "binary",
                    n_estimators=1500,
                    random_state=seed,
                    n_jobs=-1,
                    class_weight="balanced" if n_classes == 2 else None,
                ),
                {
                    "num_leaves": [31, 63, 127],
                    "learning_rate": [0.03, 0.1],
                    "min_child_samples": [10, 30, 60],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "reg_lambda": [0.0, 1.0, 5.0],
                }
            )

    results = {}
    iterator = tqdm(models.items(), total=len(models), disable=not progress, desc="Benchmark models")

    for name, (estimator, param_grid) in iterator:
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0
        )
        gs.fit(X_train, y_train)
        best = gs.best_estimator_

        y_pred = best.predict(X_test)
        y_score = _get_scores_for_auc(best, X_test, n_classes)

        results[name] = {
            "best_params": gs.best_params_,
            "cv_best_balanced_acc": float(gs.best_score_),
            "test_metrics": classification_metrics(y_test, y_pred, y_score=y_score)
        }

        if progress:
            iterator.set_postfix({
                "model": name,
                "cv_bal_acc": f"{gs.best_score_:.3f}",
                "test_bal_acc": f"{results[name]['test_metrics']['balanced_accuracy']:.3f}"
            })

    return results


# ---- Example ----
# results = run_classical_benchmarks(X_train, y_train, X_test, y_test, seed=3, cv_splits=5)
# for k, v in results.items():
#     print("\n", k)
#     print(" best_params:", v["best_params"])
#     print(" cv_best_balanced_acc:", v["cv_best_balanced_acc"])
#     print(" test_metrics:", v["test_metrics"])