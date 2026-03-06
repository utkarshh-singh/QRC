import numpy as np
from dataclasses import dataclass, asdict

from sklearn.datasets import (
    load_breast_cancer, load_wine, load_iris, load_diabetes, load_digits,
    make_classification, make_regression,
    fetch_openml
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression


@dataclass
class DatasetInfo:
    name: str
    task: str                          # "classification" or "regression"
    source: str                        # "sklearn" | "openml" | "synthetic"
    n_samples_requested: int
    n_features_requested: int
    n_samples_returned: int
    n_features_returned: int
    n_classes: int | None
    feature_transform: str             # "none", "selectkbest", "pca", "synthetic"
    scaler: str                        # "standard" or "none"
    split: str                         # "none" or "train_test_split"
    test_size: float | None
    random_state: int
    stratified: bool | None
    y_type: str
    X_dtype: str
    notes: str = ""


class CleanNumericDatasets:
    """
    Clean, numeric datasets for benchmarking.
    - Supports: sklearn built-ins, OpenML datasets, and hard synthetic datasets.
    - No leakage: feature selection/PCA/scaling fit on TRAIN ONLY when return_split=True.

    Usage:
      ds = CleanNumericDatasets(seed=7, scale=True)

      # OpenML (by name or id)
      Xtr, Xte, ytr, yte = ds.get("openml:covertype", 20000, 20, return_split=True)
      ds.info(print_out=True)

      # Hard synthetic
      Xtr, Xte, ytr, yte = ds.get("synthetic_classification_hard", 5000, 30, return_split=True)

    OpenML naming:
      - "openml:<name>"   e.g., "openml:covertype", "openml:credit-g"
      - "openml_id:<id>"  e.g., "openml_id:1590"
    """

    def __init__(self, seed: int = 0, scale: bool = True, dtype=np.float32):
        self.seed = int(seed)
        self.scale = bool(scale)
        self.dtype = dtype
        self._last_info: DatasetInfo | None = None

        # Handy OpenML shortcuts (you can add more)
        self.openml_aliases = {
            "covertype": "covertype",
            "credit-g": "credit-g",
            "default-of-credit-card-clients": "default-of-credit-card-clients",
            "california": "california",  # note: many "california" variants exist on OpenML
            "superconduct": "superconduct",
            "mnist_784": "mnist_784",
            "fashion-mnist": "Fashion-MNIST",
        }

    def info(self, print_out: bool = False):
        if self._last_info is None:
            raise RuntimeError("No dataset generated yet. Call .get(...) first.")
        info_dict = asdict(self._last_info)
        if print_out:
            for k, v in info_dict.items():
                print(f"{k}: {v}")
        return info_dict

    def _ensure_numeric(self, X, y):
        X = np.asarray(X, dtype=self.dtype)
        y = np.asarray(y)
        return X, y

    def _subsample(self, X, y, n):
        rng = np.random.default_rng(self.seed)
        n = min(int(n), X.shape[0])
        idx = rng.choice(X.shape[0], size=n, replace=False)
        return X[idx], y[idx]

    def _make_transformer(self, feature_size, task, native_dim):
        feature_size = int(feature_size)

        if feature_size == native_dim:
            return None, "none"

        if feature_size < native_dim:
            score_func = f_classif if task == "classification" else f_regression
            selector = SelectKBest(score_func=score_func, k=feature_size)
            return selector, "selectkbest"

        # If asking > native_dim, cap via PCA
        k = min(native_dim, feature_size)
        pca = PCA(n_components=k, random_state=self.seed)
        return pca, "pca(capped)"

    def _fit_transform_train_only(self, X_train, y_train, transformer, task):
        if transformer is None:
            return X_train, None
        X_train_2 = transformer.fit_transform(X_train, y_train)
        return X_train_2.astype(self.dtype, copy=False), transformer

    def _transform_test(self, X_test, transformer):
        if transformer is None:
            return X_test
        X_test_2 = transformer.transform(X_test)
        return X_test_2.astype(self.dtype, copy=False)

    def _fit_scale_train_only(self, X_train):
        if not self.scale:
            return X_train, None, "none"
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_train)
        return Xs.astype(self.dtype, copy=False), scaler, "standard"

    def _scale_test(self, X_test, scaler):
        if scaler is None:
            return X_test
        Xs = scaler.transform(X_test)
        return Xs.astype(self.dtype, copy=False)

    def _load_openml(self, name_or_id: str):
        # name_or_id can be "openml:covertype" or "openml_id:1590"
        if name_or_id.startswith("openml:"):
            key = name_or_id.split("openml:", 1)[1].strip()
            key = self.openml_aliases.get(key, key)
            data = fetch_openml(name=key, as_frame=False)
        elif name_or_id.startswith("openml_id:"):
            did = int(name_or_id.split("openml_id:", 1)[1].strip())
            data = fetch_openml(data_id=did, as_frame=False)
        else:
            raise ValueError("OpenML datasets must be requested as 'openml:<name>' or 'openml_id:<id>'.")

        X = data.data
        y = data.target
        return X, y, data.details.get("name", "openml_dataset")

    def _coerce_openml_numeric(self, X, y):
        """
        OpenML sometimes returns strings or mixed types.
        For 'clean numeric only', we enforce numeric conversion.
        If conversion fails, raise an error (so you don't silently get junk).
        """
        X = np.asarray(X)
        # if already numeric, good
        if np.issubdtype(X.dtype, np.number):
            Xn = X.astype(self.dtype, copy=False)
        else:
            # try safe conversion
            try:
                Xn = X.astype(self.dtype)
            except Exception as e:
                raise ValueError(
                    "OpenML X contains non-numeric columns. "
                    "Either choose a numeric OpenML dataset or add encoding."
                ) from e

        y_arr = np.asarray(y)
        return Xn, y_arr

    def _infer_task(self, y):
        """
        Heuristic:
          - if y is float => regression
          - else if #unique is small relative to n => classification
        """
        y_arr = np.asarray(y)
        if np.issubdtype(y_arr.dtype, np.floating):
            return "regression"

        # if string labels -> classification
        if y_arr.dtype.kind in ("U", "S", "O"):
            return "classification"

        uniq = np.unique(y_arr)
        # typical classification has small number of unique values
        if uniq.size <= 20:
            return "classification"

        # fallback
        return "regression"

    def _encode_labels_if_needed(self, y):
        """
        For OpenML classification, y can be strings. Encode to int labels.
        """
        y_arr = np.asarray(y)
        if y_arr.dtype.kind in ("U", "S", "O"):
            uniq, inv = np.unique(y_arr, return_inverse=True)
            return inv.astype(np.int64), uniq
        return y_arr, None

    def get(self,
            name: str,
            sample_size: int,
            feature_size: int,
            return_split: bool = True,
            test_size: float = 0.2,
            stratify: bool = True):
        name = str(name).strip().lower()
        sample_size = int(sample_size)
        feature_size = int(feature_size)

        # -------------------------
        # OpenML
        # -------------------------
        if name.startswith("openml:") or name.startswith("openml_id:"):
            X, y, real_name = self._load_openml(name)
            X, y = self._coerce_openml_numeric(X, y)

            task = self._infer_task(y)
            if task == "classification":
                y, class_names = self._encode_labels_if_needed(y)
                n_classes = int(len(np.unique(y)))
            else:
                class_names = None
                n_classes = None

            # subsample (after encoding)
            X, y = self._subsample(X, y, sample_size)

            native_dim = X.shape[1]
            transformer, feat_transform = self._make_transformer(feature_size, task, native_dim)

            if return_split:
                strat = y if (task == "classification" and stratify) else None
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=test_size, random_state=self.seed, stratify=strat
                )

                Xtr, transformer = self._fit_transform_train_only(Xtr, ytr, transformer, task)
                Xte = self._transform_test(Xte, transformer)

                Xtr, scaler, scaler_name = self._fit_scale_train_only(Xtr)
                Xte = self._scale_test(Xte, scaler)

                self._last_info = DatasetInfo(
                    name=f"{name} ({real_name})",
                    task=task,
                    source="openml",
                    n_samples_requested=sample_size,
                    n_features_requested=feature_size,
                    n_samples_returned=Xtr.shape[0] + Xte.shape[0],
                    n_features_returned=Xtr.shape[1],
                    n_classes=n_classes,
                    feature_transform=feat_transform,
                    scaler=scaler_name,
                    split="train_test_split",
                    test_size=float(test_size),
                    random_state=self.seed,
                    stratified=bool(stratify) if task == "classification" else None,
                    y_type=str(np.asarray(y).dtype),
                    X_dtype=str(Xtr.dtype),
                    notes=(
                        "OpenML dataset. Enforced numeric X; labels encoded if needed. "
                        "Transforms/scaling fit on train only."
                    )
                )
                return Xtr, Xte, ytr, yte

            # no split
            if transformer is not None:
                X = transformer.fit_transform(X, y).astype(self.dtype, copy=False)

            if self.scale:
                X, _, scaler_name = self._fit_scale_train_only(X)
            else:
                scaler_name = "none"

            self._last_info = DatasetInfo(
                name=f"{name} ({real_name})",
                task=task,
                source="openml",
                n_samples_requested=sample_size,
                n_features_requested=feature_size,
                n_samples_returned=X.shape[0],
                n_features_returned=X.shape[1],
                n_classes=n_classes,
                feature_transform=feat_transform,
                scaler=scaler_name,
                split="none",
                test_size=None,
                random_state=self.seed,
                stratified=None,
                y_type=str(np.asarray(y).dtype),
                X_dtype=str(X.dtype),
                notes="OpenML dataset. No split requested."
            )
            return X, y

        # -------------------------
        # Hard synthetic presets
        # -------------------------
        if name in ("synthetic_classification_hard", "syn_clf_hard"):
            task = "classification"
            n_classes = 2
            X, y = make_classification(
                n_samples=sample_size,
                n_features=feature_size,
                n_informative=max(2, int(0.3 * feature_size)),
                n_redundant=max(0, int(0.4 * feature_size)),
                n_repeated=0,
                n_classes=n_classes,
                class_sep=0.6,   # harder
                flip_y=0.05,     # noisier labels
                weights=[0.75, 0.25],  # imbalance
                random_state=self.seed,
            )
            X, y = self._ensure_numeric(X, y)

            if return_split:
                strat = y if stratify else None
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=test_size, random_state=self.seed, stratify=strat
                )
                Xtr, scaler, scaler_name = self._fit_scale_train_only(Xtr)
                Xte = self._scale_test(Xte, scaler)

                self._last_info = DatasetInfo(
                    name=name,
                    task=task,
                    source="synthetic",
                    n_samples_requested=sample_size,
                    n_features_requested=feature_size,
                    n_samples_returned=Xtr.shape[0] + Xte.shape[0],
                    n_features_returned=Xtr.shape[1],
                    n_classes=n_classes,
                    feature_transform="synthetic(hard)",
                    scaler=scaler_name,
                    split="train_test_split",
                    test_size=float(test_size),
                    random_state=self.seed,
                    stratified=bool(stratify),
                    y_type=str(np.asarray(y).dtype),
                    X_dtype=str(Xtr.dtype),
                    notes="Hard synthetic classification: low class_sep, label noise, class imbalance."
                )
                return Xtr, Xte, ytr, yte

            if self.scale:
                X, _, scaler_name = self._fit_scale_train_only(X)
            else:
                scaler_name = "none"

            self._last_info = DatasetInfo(
                name=name, task=task, source="synthetic",
                n_samples_requested=sample_size, n_features_requested=feature_size,
                n_samples_returned=X.shape[0], n_features_returned=X.shape[1],
                n_classes=n_classes,
                feature_transform="synthetic(hard)",
                scaler=scaler_name,
                split="none",
                test_size=None,
                random_state=self.seed,
                stratified=None,
                y_type=str(np.asarray(y).dtype),
                X_dtype=str(X.dtype),
                notes="Hard synthetic classification."
            )
            return X, y

        if name in ("synthetic_regression_hard", "syn_reg_hard"):
            task = "regression"
            X, y = make_regression(
                n_samples=sample_size,
                n_features=feature_size,
                n_informative=max(2, int(0.3 * feature_size)),
                noise=20.0,  # harder (noisy targets)
                random_state=self.seed,
            )
            # add mild nonlinearity (ELM should help)
            X = np.asarray(X, dtype=self.dtype)
            y = np.asarray(y, dtype=np.float64)
            y = y + 10.0 * np.sin(X[:, 0]) + 5.0 * (X[:, 1] ** 2)
            y = y.astype(np.float64)

            if return_split:
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=test_size, random_state=self.seed
                )
                Xtr, scaler, scaler_name = self._fit_scale_train_only(Xtr)
                Xte = self._scale_test(Xte, scaler)

                self._last_info = DatasetInfo(
                    name=name,
                    task=task,
                    source="synthetic",
                    n_samples_requested=sample_size,
                    n_features_requested=feature_size,
                    n_samples_returned=Xtr.shape[0] + Xte.shape[0],
                    n_features_returned=Xtr.shape[1],
                    n_classes=None,
                    feature_transform="synthetic(hard)",
                    scaler=scaler_name,
                    split="train_test_split",
                    test_size=float(test_size),
                    random_state=self.seed,
                    stratified=None,
                    y_type=str(np.asarray(y).dtype),
                    X_dtype=str(Xtr.dtype),
                    notes="Hard synthetic regression: noisy + injected nonlinearity."
                )
                return Xtr, Xte, ytr, yte

            if self.scale:
                X, _, scaler_name = self._fit_scale_train_only(X)
            else:
                scaler_name = "none"

            self._last_info = DatasetInfo(
                name=name, task=task, source="synthetic",
                n_samples_requested=sample_size, n_features_requested=feature_size,
                n_samples_returned=X.shape[0], n_features_returned=X.shape[1],
                n_classes=None,
                feature_transform="synthetic(hard)",
                scaler=scaler_name,
                split="none",
                test_size=None,
                random_state=self.seed,
                stratified=None,
                y_type=str(np.asarray(y).dtype),
                X_dtype=str(X.dtype),
                notes="Hard synthetic regression."
            )
            return X, y

        # -------------------------
        # Existing sklearn + easy synthetic (your old behavior)
        # -------------------------
        # Keep your old names working by mapping them:
        if name in ("synthetic_classification", "syn_classification", "clf_synth"):
            # easier synthetic (your old version)
            task = "classification"
            n_classes = 2
            X, y = make_classification(
                n_samples=sample_size,
                n_features=feature_size,
                n_informative=max(2, min(feature_size, feature_size // 2)),
                n_redundant=max(0, min(feature_size - 2, feature_size // 4)),
                n_repeated=0,
                n_classes=n_classes,
                class_sep=1.0,
                flip_y=0.01,
                random_state=self.seed,
            )
            X, y = self._ensure_numeric(X, y)

            if return_split:
                strat = y if stratify else None
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=test_size, random_state=self.seed, stratify=strat
                )
                Xtr, scaler, scaler_name = self._fit_scale_train_only(Xtr)
                Xte = self._scale_test(Xte, scaler)

                self._last_info = DatasetInfo(
                    name=name, task=task, source="synthetic",
                    n_samples_requested=sample_size, n_features_requested=feature_size,
                    n_samples_returned=Xtr.shape[0] + Xte.shape[0],
                    n_features_returned=Xtr.shape[1],
                    n_classes=n_classes,
                    feature_transform="synthetic",
                    scaler=scaler_name,
                    split="train_test_split",
                    test_size=float(test_size),
                    random_state=self.seed,
                    stratified=bool(stratify),
                    y_type=str(np.asarray(y).dtype),
                    X_dtype=str(Xtr.dtype),
                    notes="Synthetic classification via make_classification; scaling fit on train only."
                )
                return Xtr, Xte, ytr, yte

            if self.scale:
                X, _, scaler_name = self._fit_scale_train_only(X)
            else:
                scaler_name = "none"

            self._last_info = DatasetInfo(
                name=name, task=task, source="synthetic",
                n_samples_requested=sample_size, n_features_requested=feature_size,
                n_samples_returned=X.shape[0], n_features_returned=X.shape[1],
                n_classes=n_classes,
                feature_transform="synthetic",
                scaler=scaler_name,
                split="none",
                test_size=None,
                random_state=self.seed,
                stratified=None,
                y_type=str(np.asarray(y).dtype),
                X_dtype=str(X.dtype),
                notes="Synthetic classification."
            )
            return X, y

        if name in ("synthetic_regression", "syn_regression", "reg_synth"):
            task = "regression"
            X, y = make_regression(
                n_samples=sample_size,
                n_features=feature_size,
                n_informative=max(2, min(feature_size, feature_size // 2)),
                noise=0.1,
                random_state=self.seed,
            )
            X, y = self._ensure_numeric(X, y)

            if return_split:
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=test_size, random_state=self.seed
                )
                Xtr, scaler, scaler_name = self._fit_scale_train_only(Xtr)
                Xte = self._scale_test(Xte, scaler)

                self._last_info = DatasetInfo(
                    name=name, task=task, source="synthetic",
                    n_samples_requested=sample_size, n_features_requested=feature_size,
                    n_samples_returned=Xtr.shape[0] + Xte.shape[0],
                    n_features_returned=Xtr.shape[1],
                    n_classes=None,
                    feature_transform="synthetic",
                    scaler=scaler_name,
                    split="train_test_split",
                    test_size=float(test_size),
                    random_state=self.seed,
                    stratified=None,
                    y_type=str(np.asarray(y).dtype),
                    X_dtype=str(Xtr.dtype),
                    notes="Synthetic regression via make_regression; scaling fit on train only."
                )
                return Xtr, Xte, ytr, yte

            if self.scale:
                X, _, scaler_name = self._fit_scale_train_only(X)
            else:
                scaler_name = "none"

            self._last_info = DatasetInfo(
                name=name, task=task, source="synthetic",
                n_samples_requested=sample_size, n_features_requested=feature_size,
                n_samples_returned=X.shape[0], n_features_returned=X.shape[1],
                n_classes=None,
                feature_transform="synthetic",
                scaler=scaler_name,
                split="none",
                test_size=None,
                random_state=self.seed,
                stratified=None,
                y_type=str(np.asarray(y).dtype),
                X_dtype=str(X.dtype),
                notes="Synthetic regression."
            )
            return X, y

        # sklearn built-ins (same as your previous version, but with no leakage if split)
        if name == "breast_cancer":
            data = load_breast_cancer()
            X, y = data.data, data.target
            task = "classification"
            real_name = "breast_cancer"
        elif name == "wine":
            data = load_wine()
            X, y = data.data, data.target
            task = "classification"
            real_name = "wine"
        elif name == "iris":
            data = load_iris()
            X, y = data.data, data.target
            task = "classification"
            real_name = "iris"
        elif name == "digits":
            data = load_digits()
            X, y = data.data, data.target
            task = "classification"
            real_name = "digits"
        elif name == "diabetes":
            data = load_diabetes()
            X, y = data.data, data.target
            task = "regression"
            real_name = "diabetes"
        else:
            raise ValueError(
                f"Unknown dataset name '{name}'. Try: "
                f"breast_cancer, wine, iris, digits, diabetes, "
                f"synthetic_classification, synthetic_regression, "
                f"synthetic_classification_hard, synthetic_regression_hard, "
                f"openml:<name> or openml_id:<id>"
            )

        X, y = self._ensure_numeric(X, y)
        X, y = self._subsample(X, y, sample_size)

        n_classes = int(len(np.unique(y))) if task == "classification" else None
        native_dim = X.shape[1]
        transformer, feat_transform = self._make_transformer(feature_size, task, native_dim)

        if return_split:
            strat = y if (task == "classification" and stratify) else None
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=test_size, random_state=self.seed, stratify=strat
            )

            Xtr, transformer = self._fit_transform_train_only(Xtr, ytr, transformer, task)
            Xte = self._transform_test(Xte, transformer)

            Xtr, scaler, scaler_name = self._fit_scale_train_only(Xtr)
            Xte = self._scale_test(Xte, scaler)

            self._last_info = DatasetInfo(
                name=real_name, task=task, source="sklearn",
                n_samples_requested=sample_size, n_features_requested=feature_size,
                n_samples_returned=Xtr.shape[0] + Xte.shape[0],
                n_features_returned=Xtr.shape[1],
                n_classes=n_classes,
                feature_transform=feat_transform,
                scaler=scaler_name,
                split="train_test_split",
                test_size=float(test_size),
                random_state=self.seed,
                stratified=bool(stratify) if task == "classification" else None,
                y_type=str(np.asarray(y).dtype),
                X_dtype=str(Xtr.dtype),
                notes="sklearn dataset; transforms/scaling fit on train only."
            )
            return Xtr, Xte, ytr, yte

        # no split
        if transformer is not None:
            X = transformer.fit_transform(X, y).astype(self.dtype, copy=False)

        if self.scale:
            X, _, scaler_name = self._fit_scale_train_only(X)
        else:
            scaler_name = "none"

        self._last_info = DatasetInfo(
            name=real_name, task=task, source="sklearn",
            n_samples_requested=sample_size, n_features_requested=feature_size,
            n_samples_returned=X.shape[0],
            n_features_returned=X.shape[1],
            n_classes=n_classes,
            feature_transform=feat_transform,
            scaler=scaler_name,
            split="none",
            test_size=None,
            random_state=self.seed,
            stratified=None,
            y_type=str(np.asarray(y).dtype),
            X_dtype=str(X.dtype),
            notes="sklearn dataset; no split requested."
        )
        return X, y