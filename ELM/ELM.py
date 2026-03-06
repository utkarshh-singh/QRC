import numpy as np
from tqdm import tqdm

from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler

from utils import ReservoirWrapper


class QuantumELM:
    """
    Extreme Learning Machine (ELM) version of the quantum reservoir:
    - NO feedback loop
    - NO recurrent state
    - Each sample x -> reservoir.compute(x) -> fixed nonlinear features
    - Train ONLY a readout (can be regression-style or proper classifier readout)

    Classification handling:
      - If model_type is a proper classifier readout: (logreg, ridge_classifier, svc)
            -> predict() returns class labels; threshold is NOT used.
      - Otherwise (ridge/lasso/linear/svr):
            -> if y is binary (2 unique integer labels), predict() applies threshold to scores.
               Use threshold="auto" (default) or a float value.

    Improvements (keeps old functionality):
      - Optional feature scaling on Phi(X): scale_features=True by default
      - Optional Ridge solver choice: ridge_solver="svd" by default (stable)
      - Optional parallel feature extraction:
            parallel=True/False, n_jobs (int), backend="threads"|"processes"
        NOTE: Use threads if your reservoir.compute releases the GIL (often true if heavy in numpy/Qiskit),
              otherwise processes can help but require picklable reservoir.
    """

    _CLASSIFIER_READOUTS = {"logreg", "logistic", "logisticregression",
                           "ridge_classifier", "ridgeclassifier", "svc"}

    def __init__(self,
                 reservoir,
                 regularization=1e-6,
                 show_progress=True,
                 model_type="ridge",
                 limit=None,
                 save_states=True,
                 threshold="auto",
                 dtype=np.float32,
                 scale_features=True,
                 ridge_solver="svd",
                 parallel=False,
                 n_jobs=-1,
                 backend="threads",
                 chunk_size=32):
        self.reservoir = ReservoirWrapper(reservoir)
        self.regularization = float(regularization)
        self.show_progress = show_progress
        self.model_type = str(model_type).lower()
        self.limit = limit
        self.save_states = save_states
        self.threshold_mode = threshold  # "auto" or float
        self.dtype = dtype

        self.scale_features = bool(scale_features)
        self.ridge_solver = ridge_solver

        # parallel options (optional)
        self.parallel = bool(parallel)
        self.n_jobs = int(n_jobs)
        self.backend = str(backend).lower()   # "threads" or "processes"
        self.chunk_size = int(chunk_size)

        self.model = self._initialize_model()

        # learned in fit()
        self.is_binary_classification_ = False
        self.classes_ = None
        self.threshold_ = None

        # feature cache
        self.saved_features_ = None

        # scaler for Phi(X)
        self.phi_scaler_ = None

    def _initialize_model(self):
        mt = self.model_type
        models = {
            "ridge": Ridge(alpha=self.regularization, solver=self.ridge_solver),
            "lasso": Lasso(alpha=self.regularization),
            "linear": LinearRegression(),
            "svr": SVR(),
            "svc": SVC(),  # classifier readout
            "logreg": LogisticRegression(max_iter=2000),
            "logistic": LogisticRegression(max_iter=2000),
            "logisticregression": LogisticRegression(max_iter=2000),
            "ridge_classifier": RidgeClassifier(alpha=self.regularization),
            "ridgeclassifier": RidgeClassifier(alpha=self.regularization),
        }
        if mt not in models:
            raise ValueError(
                "Invalid model_type. Choose from: "
                "'ridge', 'lasso', 'linear', 'svr', 'svc', 'logreg', 'ridge_classifier'."
            )
        return models[mt]

    def _is_classifier_readout(self):
        return self.model_type in self._CLASSIFIER_READOUTS

    def _process_features(self, quantum_state):
        q = np.asarray(quantum_state).flatten().astype(self.dtype, copy=False)
        if self.limit is None:
            return q
        out_len = int(self.limit * len(q))
        out_len = max(1, out_len)
        return q[:out_len]

    def _compute_one(self, x):
        x = np.asarray(x, dtype=self.dtype)
        q_state = self.reservoir.compute(x)  # <-- NO feedback applied
        return self._process_features(q_state)

    def _maybe_fit_scale_phi(self, Phi):
        if not self.scale_features:
            self.phi_scaler_ = None
            return Phi
        self.phi_scaler_ = StandardScaler()
        Phi_s = self.phi_scaler_.fit_transform(Phi)
        return Phi_s.astype(self.dtype, copy=False)

    def _maybe_scale_phi(self, Phi):
        if (not self.scale_features) or (self.phi_scaler_ is None):
            return Phi
        Phi_s = self.phi_scaler_.transform(Phi)
        return Phi_s.astype(self.dtype, copy=False)

    def transform(self, X, load_saved=False):
        """
        Compute ELM features Phi(X). Returns RAW Phi (unscaled).
        Scaling is applied in fit/predict.
        """
        if load_saved and self.saved_features_ is not None:
            return self.saved_features_

        X = np.asarray(X)

        # --- serial path ---
        if not self.parallel:
            feats = []
            iterator = tqdm(X, desc="ELM feature extraction", unit=" sample",
                            disable=not self.show_progress)
            for x in iterator:
                feats.append(self._compute_one(x))
            feats = np.asarray(feats, dtype=self.dtype)

        # --- parallel path ---
        else:
            from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

            Executor = ThreadPoolExecutor if self.backend in ("threads", "thread") else ProcessPoolExecutor

            feats = [None] * len(X)
            iterator = range(len(X))
            pbar = tqdm(total=len(X), desc="ELM feature extraction (parallel)",
                        unit=" sample", disable=not self.show_progress)

            # IMPORTANT:
            # - ThreadPool works fine even if reservoir isn't picklable.
            # - ProcessPool requires everything used in _compute_one to be picklable.
            with Executor(max_workers=None if self.n_jobs == -1 else self.n_jobs) as ex:
                futures = {}

                # chunked submission to reduce overhead
                for start in range(0, len(X), self.chunk_size):
                    end = min(len(X), start + self.chunk_size)
                    for i in range(start, end):
                        futures[ex.submit(self._compute_one, X[i])] = i

                for fut in as_completed(futures):
                    i = futures[fut]
                    feats[i] = fut.result()
                    pbar.update(1)

            pbar.close()
            feats = np.asarray(feats, dtype=self.dtype)

        if self.save_states:
            self.saved_features_ = feats

        return feats

    def fit(self, X, y, load_saved=False):
        y = np.asarray(y)
        uniq = np.unique(y)

        self.is_binary_classification_ = (uniq.size == 2) and np.issubdtype(y.dtype, np.integer)
        self.classes_ = uniq if self.is_binary_classification_ else None

        Phi = self.transform(X, load_saved=load_saved)
        Phi = self._maybe_fit_scale_phi(Phi)

        self.model.fit(Phi, y)

        self.threshold_ = None
        if self.is_binary_classification_ and (not self._is_classifier_readout()):
            if self.threshold_mode == "auto":
                if set(uniq.tolist()) == {0, 1}:
                    self.threshold_ = 0.5
                elif set(uniq.tolist()) == {-1, 1}:
                    self.threshold_ = 0.0
                else:
                    self.threshold_ = float(np.mean(uniq))
            else:
                self.threshold_ = float(self.threshold_mode)

        return self

    def predict_scores(self, X, load_saved=False):
        Phi = self.transform(X, load_saved=load_saved)
        Phi = self._maybe_scale_phi(Phi)

        if self._is_classifier_readout():
            if hasattr(self.model, "predict_proba") and self.is_binary_classification_:
                proba = self.model.predict_proba(Phi)
                return proba[:, 1]
            if hasattr(self.model, "decision_function"):
                return self.model.decision_function(Phi)
            return self.model.predict(Phi)

        return self.model.predict(Phi)

    def predict(self, X, load_saved=False):
        Phi = self.transform(X, load_saved=load_saved)
        Phi = self._maybe_scale_phi(Phi)

        if self._is_classifier_readout():
            return self.model.predict(Phi)

        pred = self.model.predict(Phi)

        if self.is_binary_classification_:
            thr = 0.5 if self.threshold_ is None else self.threshold_
            y_bin01 = (pred >= thr).astype(int)
            if self.classes_ is not None and set(self.classes_.tolist()) == {-1, 1}:
                return y_bin01 * 2 - 1
            return y_bin01

        return pred

    def get_saved_features(self):
        if self.saved_features_ is None:
            raise ValueError("No saved features available. Run transform() or fit() first.")
        return self.saved_features_