"""
advanced_methods.py — Kernel Ridge & Bayesian Linear Regression from scratch
==========================================================================
Beijing PM2.5 Air Quality Dataset (UCI #501)
FIT – HCMUS | Toán Ứng Dụng và Thống Kê | Đồ án 2

All algorithms custom-written from scratch using NumPy.
"""

import numpy as np
from scipy import stats
from model_comparison import (
    BaseModel,
    _add_intercept,
    compute_mae,
    compute_rmse,
    compute_r2,
    compute_adjusted_r2
)

# ── Reproducibility ──────────────────────────────────────────
SEED = 42


# ═══════════════════════════════════════════════════════════════
#  1. KERNEL RIDGE REGRESSION
# ═══════════════════════════════════════════════════════════════

class KernelRidgeModel(BaseModel):
    """
    Kernel Ridge Regression (KRR) with RBF kernel — from scratch.
    ŷ(x) = k(x)ᵀ (K + λI)⁻¹ y
    """
    name = "Kernel Ridge"

    def __init__(self, lam: float = 1.0, length_scale: float = 1.0, subsample_limit: int = 3000):
        self.lam = lam
        self.length_scale = length_scale
        self.subsample_limit = subsample_limit
        self.alpha: np.ndarray | None = None
        self.X_train: np.ndarray | None = None
        self._n_features: int = 0

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        RBF (Gaussian) Kernel: K(x, x') = exp(-||x-x'||² / 2ℓ²)
        Uses optimized vectorization to prevent memory overhead.
        """
        sq_X1 = np.sum(X1**2, axis=1, keepdims=True)
        sq_X2 = np.sum(X2**2, axis=1, keepdims=True)
        dists = sq_X1 + sq_X2.T - 2.0 * X1 @ X2.T
        dists = np.clip(dists, 0.0, None)  # Clip negative values due to numerical precision
        return np.exp(-dists / (2.0 * self.length_scale**2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelRidgeModel":
        self._n_features = X.shape[1]
        n = X.shape[0]

        # Memory Optimization: Subsample if n exceeds subsample_limit
        if n > self.subsample_limit:
            np.random.seed(SEED)
            indices = np.random.choice(n, self.subsample_limit, replace=False)
            self.X_train = X[indices].copy()
            y_train = y[indices].copy()
        else:
            self.X_train = X.copy()
            y_train = y.copy()

        K = self._rbf_kernel(self.X_train, self.X_train)
        n_samples = len(self.X_train)

        # (K + λI) α = y
        try:
            self.alpha = np.linalg.solve(K + self.lam * np.eye(n_samples), y_train)
        except np.linalg.LinAlgError:
            self.alpha = np.linalg.lstsq(K + self.lam * np.eye(n_samples), y_train, rcond=1e-15)[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.alpha is None or self.X_train is None:
            raise ValueError("Model is not fitted yet!")
        K_test = self._rbf_kernel(X, self.X_train)
        return K_test @ self.alpha

    def get_coefficients(self) -> np.ndarray:
        # Return dual coefficients instead of primal weights
        return self.alpha

    def get_intercept(self) -> float:
        return 0.0

    def summary(self) -> dict:
        return {
            "name": self.name,
            "lambda": self.lam,
            "length_scale": self.length_scale,
            "n_train_subsampled": len(self.X_train) if self.X_train is not None else 0
        }


# ═══════════════════════════════════════════════════════════════
#  2. BAYESIAN LINEAR REGRESSION
# ═══════════════════════════════════════════════════════════════

class BayesianLinearModel(BaseModel):
    """
    Bayesian Linear Regression with conjugate normal-inverse-gamma prior.
    Calculates posterior mean (mn) and posterior covariance (Sn) from scratch.
    Enables uncertainty quantification (predictive standard deviation).
    """
    name = "Bayesian Linear"

    def __init__(self, alpha: float = 1.0, sigma2: float = 1.0):
        self.alpha = alpha  # Prior precision (1 / variance of weights prior)
        self.sigma2 = sigma2  # Initial noise variance estimate
        self.beta_hat: np.ndarray | None = None
        self.S_n: np.ndarray | None = None
        self.m_n: np.ndarray | None = None
        self._n_features: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLinearModel":
        self._n_features = X.shape[1]
        X_b = _add_intercept(X)
        n, p = X_b.shape

        # Prior: β ~ N(0, (1/α)I)
        S0_inv = self.alpha * np.eye(p)
        m0 = np.zeros(p)

        # First pass: Fit model and estimate sigma2 from residuals
        XtX = X_b.T @ X_b
        Xty = X_b.T @ y
        
        Sn_inv = S0_inv + (1.0 / self.sigma2) * XtX
        try:
            self.S_n = np.linalg.inv(Sn_inv)
        except np.linalg.LinAlgError:
            self.S_n = np.linalg.pinv(Sn_inv)
        self.m_n = self.S_n @ (S0_inv @ m0 + (1.0 / self.sigma2) * Xty)

        # Estimate actual noise variance from the data
        residuals = y - X_b @ self.m_n
        self.sigma2 = float(np.sum(residuals ** 2) / (n - p))

        # Second pass: Re-calculate posterior parameters with estimated noise variance
        Sn_inv = S0_inv + (1.0 / self.sigma2) * XtX
        try:
            self.S_n = np.linalg.inv(Sn_inv)
        except np.linalg.LinAlgError:
            self.S_n = np.linalg.pinv(Sn_inv)
        self.m_n = self.S_n @ (S0_inv @ m0 + (1.0 / self.sigma2) * Xty)

        self.beta_hat = self.m_n  # MAP (Maximum A Posteriori) estimate
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.m_n is None:
            raise ValueError("Model is not fitted yet!")
        X_b = _add_intercept(X)
        return X_b @ self.m_n

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns predictions along with posterior predictive standard deviation.
        σ²_pred(x) = σ² + xᵀ S_n x
        """
        if self.m_n is None or self.S_n is None:
            raise ValueError("Model is not fitted yet!")
        X_b = _add_intercept(X)
        y_mean = X_b @ self.m_n
        # Calculate predictive variance for each test sample
        y_var = self.sigma2 + np.sum((X_b @ self.S_n) * X_b, axis=1)
        y_std = np.sqrt(y_var)
        return y_mean, y_std

    def get_coefficients(self) -> np.ndarray:
        return self.beta_hat[1:]

    def get_intercept(self) -> float:
        return float(self.beta_hat[0])

    def summary(self) -> dict:
        return {
            "name": self.name,
            "alpha": self.alpha,
            "noise_variance": self.sigma2,
            "n_features": self._n_features
        }


# ═══════════════════════════════════════════════════════════════
#  3. UNIT TESTS FOR ADVANCED METHODS
# ═══════════════════════════════════════════════════════════════

def _run_tests():
    """Unit tests for advanced models. ≥ 2 tests per function/class."""
    print("\n" + "=" * 60)
    print("  RUNNING UNIT TESTS FOR ADVANCED METHODS")
    print("=" * 60)
    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}")
            failed += 1

    np.random.seed(SEED)
    X_syn = np.random.randn(200, 2)
    y_syn = 2.0 * X_syn[:, 0] - X_syn[:, 1] + 3.0 + 0.1 * np.random.randn(200)

    # ── Kernel Ridge Model tests ─────────────────────────────
    krr = KernelRidgeModel(lam=0.1, length_scale=1.5)
    krr.fit(X_syn, y_syn)
    
    check("Kernel Ridge: fit", krr.alpha is not None)
    
    y_pred_krr = krr.predict(X_syn)
    r2_krr = compute_r2(y_syn, y_pred_krr)
    check("Kernel Ridge: accuracy (R² > 0.9)", r2_krr > 0.9)
    check("Kernel Ridge: subsample logic", krr.X_train.shape[0] == 200)

    # ── Bayesian Linear Model tests ──────────────────────────
    bayes = BayesianLinearModel(alpha=1.0)
    bayes.fit(X_syn, y_syn)
    
    check("Bayesian Linear: posterior mean", bayes.m_n is not None)
    check("Bayesian Linear: posterior covariance", bayes.S_n.shape == (3, 3))
    
    y_mean, y_std = bayes.predict_with_uncertainty(X_syn)
    check("Bayesian Linear: mean predictions", y_mean is not None)
    check("Bayesian Linear: uncertainty intervals > 0", np.all(y_std > 0))
    
    r2_bayes = compute_r2(y_syn, y_mean)
    check("Bayesian Linear: accuracy (R² > 0.9)", r2_bayes > 0.9)

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 60 + "\n")
    return failed == 0


if __name__ == "__main__":
    _run_tests()
