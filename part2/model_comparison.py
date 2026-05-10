"""
model_comparison.py — Regression Model Builder & Comparator
============================================================
Beijing PM2.5 Air Quality Dataset (UCI #501)
FIT – HCMUS | Toán Ứng Dụng và Thống Kê | Đồ án 2

All core algorithms implemented from scratch using NumPy.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Figures directory ────────────────────────────────────────
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  1. EVALUATION METRICS (from scratch)
# ═══════════════════════════════════════════════════════════════

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE = (1/n) × Σ|yᵢ − ŷᵢ|"""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE = √[(1/n) × Σ(yᵢ − ŷᵢ)²]"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² = 1 − RSS/TSS = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²"""
    rss = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    if tss == 0:
        return 0.0
    return float(1.0 - rss / tss)


def compute_adjusted_r2(y_true: np.ndarray, y_pred: np.ndarray, p: int) -> float:
    """Adjusted R² = 1 − (1−R²)(n−1)/(n−p−1)"""
    n = len(y_true)
    r2 = compute_r2(y_true, y_pred)
    if n - p - 1 <= 0:
        return r2
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))


def _add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones for the intercept term."""
    return np.column_stack([np.ones(X.shape[0]), X])


# ═══════════════════════════════════════════════════════════════
#  2. BASE MODEL (Abstract Interface)
# ═══════════════════════════════════════════════════════════════

class BaseModel:
    """
    Abstract base class for all regression models.
    Uniform interface enables pluggable model comparison.
    Advanced models (Kernel Ridge, Bayesian) will inherit this same interface.
    """
    name: str = "BaseModel"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_coefficients(self) -> np.ndarray:
        raise NotImplementedError

    def get_intercept(self) -> float:
        raise NotImplementedError

    def summary(self) -> dict:
        raise NotImplementedError

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Compute all metrics on a dataset."""
        y_pred = self.predict(X)
        p = X.shape[1]
        return {
            "MAE": compute_mae(y, y_pred),
            "RMSE": compute_rmse(y, y_pred),
            "R2": compute_r2(y, y_pred),
            "Adj_R2": compute_adjusted_r2(y, y_pred, p),
        }


# ═══════════════════════════════════════════════════════════════
#  3. OLS BASIC MODEL
# ═══════════════════════════════════════════════════════════════

class OLSBasicModel(BaseModel):
    """
    Ordinary Least Squares — closed-form Normal Equation.
    β̂ = (XᵀX)⁻¹Xᵀy
    """
    name = "OLS Basic"

    def __init__(self):
        self.beta_hat: np.ndarray | None = None
        self.sigma2: float | None = None
        self._n_features: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OLSBasicModel":
        X_b = _add_intercept(X)
        n, p_full = X_b.shape
        self._n_features = X.shape[1]

        # β̂ = (XᵀX)⁻¹Xᵀy — solved via LU decomposition for stability
        XtX = X_b.T @ X_b
        Xty = X_b.T @ y
        self.beta_hat = np.linalg.solve(XtX, Xty)

        # σ̂² = RSS / (n − p − 1)
        residuals = y - X_b @ self.beta_hat
        self.sigma2 = float(np.sum(residuals ** 2) / (n - p_full))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = _add_intercept(X)
        return X_b @ self.beta_hat

    def get_coefficients(self) -> np.ndarray:
        return self.beta_hat[1:]

    def get_intercept(self) -> float:
        return float(self.beta_hat[0])

    def inference(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Statistical inference: standard errors, t-stats, p-values, 95% CI.
        """
        X_b = _add_intercept(X)
        n, p_full = X_b.shape
        df = n - p_full  # degrees of freedom

        # Variance-covariance matrix: σ̂²(XᵀX)⁻¹
        XtX_inv = np.linalg.inv(X_b.T @ X_b)
        var_cov = self.sigma2 * XtX_inv

        # Standard errors: se(β̂ⱼ) = √[Var(β̂ⱼ)]
        std_errors = np.sqrt(np.diag(var_cov))

        # t-statistics: tⱼ = β̂ⱼ / se(β̂ⱼ)
        t_stats = self.beta_hat / std_errors

        # p-values: pⱼ = 2 × P(T > |tⱼ|) — two-tailed test
        p_values = 2.0 * stats.t.sf(np.abs(t_stats), df)

        # 95% Confidence intervals: β̂ⱼ ± t_{α/2, df} × se(β̂ⱼ)
        t_crit = stats.t.ppf(0.975, df)
        ci_lower = self.beta_hat - t_crit * std_errors
        ci_upper = self.beta_hat + t_crit * std_errors

        return {
            "std_errors": std_errors,
            "t_stats": t_stats,
            "p_values": p_values,
            "ci_95": np.column_stack([ci_lower, ci_upper]),
        }

    def summary(self) -> dict:
        return {"name": self.name, "n_features": self._n_features}


# ═══════════════════════════════════════════════════════════════
#  4. OLS WITH FEATURE SELECTION (Backward Elimination)
# ═══════════════════════════════════════════════════════════════

def compute_vif(X: np.ndarray) -> np.ndarray:
    """
    Variance Inflation Factor for each feature.
    VIF(j) = 1 / (1 − R²_j)
    where R²_j is obtained by regressing feature j on all other features.
    """
    p = X.shape[1]
    vif_values = np.zeros(p)
    for j in range(p):
        # Regress X[:,j] on all other columns
        mask = np.ones(p, dtype=bool)
        mask[j] = False
        X_other = X[:, mask]
        y_j = X[:, j]

        X_b = _add_intercept(X_other)
        beta = np.linalg.solve(X_b.T @ X_b, X_b.T @ y_j)
        y_hat = X_b @ beta
        r2_j = compute_r2(y_j, y_hat)

        vif_values[j] = 1.0 / (1.0 - r2_j) if r2_j < 1.0 else np.inf
    return vif_values


class OLSFeatureSelectedModel(BaseModel):
    """
    OLS with Backward Elimination based on p-value and VIF.
    Iteratively removes the worst feature until all pass thresholds.
    """
    name = "OLS Feature Selected"

    def __init__(self, significance_level: float = 0.05, vif_threshold: float = 10.0):
        self.significance_level = significance_level
        self.vif_threshold = vif_threshold
        self.selected_features: list[int] = []
        self.elimination_history: list[dict] = []
        self._ols: OLSBasicModel | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OLSFeatureSelectedModel":
        p = X.shape[1]
        remaining = list(range(p))
        self.elimination_history = []

        while True:
            X_sub = X[:, remaining]

            # Fit OLS on current feature set
            ols = OLSBasicModel()
            ols.fit(X_sub, y)
            inf = ols.inference(X_sub, y)

            # p-values for feature coefficients (skip intercept at index 0)
            p_vals = inf["p_values"][1:]

            # Check VIF
            vif_vals = compute_vif(X_sub) if len(remaining) > 1 else np.zeros(1)

            # Find worst feature by p-value
            worst_p_idx = int(np.argmax(p_vals))
            worst_p_val = p_vals[worst_p_idx]

            # Find worst feature by VIF
            worst_vif_idx = int(np.argmax(vif_vals))
            worst_vif_val = vif_vals[worst_vif_idx]

            # Decide which to remove (prioritize VIF > threshold)
            remove_idx = None
            reason = ""
            if worst_vif_val > self.vif_threshold:
                remove_idx = worst_vif_idx
                reason = f"VIF={worst_vif_val:.2f} > {self.vif_threshold}"
            elif worst_p_val > self.significance_level:
                remove_idx = worst_p_idx
                reason = f"p-value={worst_p_val:.4f} > {self.significance_level}"

            if remove_idx is None:
                break  # All features pass both criteria

            removed_feat = remaining.pop(remove_idx)
            self.elimination_history.append({
                "step": len(self.elimination_history) + 1,
                "removed_feature_idx": removed_feat,
                "reason": reason,
                "remaining_count": len(remaining),
            })

            if len(remaining) == 0:
                break

        self.selected_features = remaining
        self._ols = OLSBasicModel()
        self._ols.fit(X[:, self.selected_features], y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._ols.predict(X[:, self.selected_features])

    def get_coefficients(self) -> np.ndarray:
        return self._ols.get_coefficients()

    def get_intercept(self) -> float:
        return self._ols.get_intercept()

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_features_original": len(self.selected_features) + len(self.elimination_history),
            "n_features_selected": len(self.selected_features),
            "selected_features": self.selected_features,
            "elimination_history": self.elimination_history,
        }


# ═══════════════════════════════════════════════════════════════
#  5. RIDGE REGRESSION
# ═══════════════════════════════════════════════════════════════

class RidgeModel(BaseModel):
    """
    Ridge Regression — L2 regularization.
    β̂_ridge = (XᵀX + λI)⁻¹Xᵀy
    λ controls the strength of the penalty (chosen via cross-validation).
    """
    name = "Ridge"

    def __init__(self, lam: float = 1.0):
        self.lam = lam
        self.beta_hat: np.ndarray | None = None
        self._n_features: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeModel":
        X_b = _add_intercept(X)
        n, p_full = X_b.shape
        self._n_features = X.shape[1]

        XtX = X_b.T @ X_b
        Xty = X_b.T @ y

        # I_mod: do NOT penalize the intercept (first column)
        I_mod = np.eye(p_full)
        I_mod[0, 0] = 0.0

        # β̂_ridge = (XᵀX + λI_mod)⁻¹Xᵀy
        self.beta_hat = np.linalg.solve(XtX + self.lam * I_mod, Xty)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = _add_intercept(X)
        return X_b @ self.beta_hat

    def get_coefficients(self) -> np.ndarray:
        return self.beta_hat[1:]

    def get_intercept(self) -> float:
        return float(self.beta_hat[0])

    def summary(self) -> dict:
        return {"name": self.name, "lambda": self.lam, "n_features": self._n_features}


# ═══════════════════════════════════════════════════════════════
#  6. LASSO REGRESSION (Coordinate Descent from scratch)
# ═══════════════════════════════════════════════════════════════

def _soft_threshold(z: float, gamma: float) -> float:
    """
    Soft-thresholding operator: S(z, γ) = sign(z) × max(|z| − γ, 0)
    WHY: The L1 penalty makes the Lasso objective non-differentiable.
    The soft-threshold operator is the proximal operator for the L1 norm.
    """
    if z > gamma:
        return z - gamma
    elif z < -gamma:
        return z + gamma
    else:
        return 0.0


class LassoModel(BaseModel):
    """
    Lasso Regression — L1 regularization via Coordinate Descent.
    Minimizes: (1/2n)||y − Xβ||² + λ||β||₁
    Uses soft-thresholding to handle non-differentiability of L1 penalty.
    """
    name = "Lasso"

    def __init__(self, lam: float = 1.0, max_iter: int = 1000, tol: float = 1e-4):
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.beta_hat: np.ndarray | None = None
        self._n_features: int = 0
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoModel":
        X_b = _add_intercept(X)
        n, p_full = X_b.shape
        self._n_features = X.shape[1]

        # Initialize β with zeros
        beta = np.zeros(p_full)
        # Pre-compute column norms (squared) for efficiency
        col_norms_sq = np.sum(X_b ** 2, axis=0) / n

        for iteration in range(self.max_iter):
            beta_old = beta.copy()

            for j in range(p_full):
                # Partial residual: exclude contribution of feature j
                r_j = y - X_b @ beta + X_b[:, j] * beta[j]

                # Unnormalized update: z_j = (1/n) × Xⱼᵀrⱼ
                z_j = X_b[:, j] @ r_j / n

                if j == 0:
                    # Don't regularize the intercept
                    beta[j] = z_j / col_norms_sq[j] if col_norms_sq[j] > 0 else 0.0
                else:
                    # Apply soft-thresholding for L1 penalty
                    beta[j] = _soft_threshold(z_j, self.lam) / col_norms_sq[j] if col_norms_sq[j] > 0 else 0.0

            # Check convergence: max absolute change in coefficients
            if np.max(np.abs(beta - beta_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.beta_hat = beta
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = _add_intercept(X)
        return X_b @ self.beta_hat

    def get_coefficients(self) -> np.ndarray:
        return self.beta_hat[1:]

    def get_intercept(self) -> float:
        return float(self.beta_hat[0])

    def get_zero_coefficients(self, feature_names: list[str] | None = None) -> list:
        """Return features with coefficient exactly zero (sparsity from L1)."""
        coefs = self.get_coefficients()
        zero_mask = np.abs(coefs) < 1e-10
        if feature_names is not None:
            return [feature_names[i] for i in range(len(coefs)) if zero_mask[i]]
        return list(np.where(zero_mask)[0])

    def summary(self) -> dict:
        coefs = self.get_coefficients()
        n_zero = int(np.sum(np.abs(coefs) < 1e-10))
        return {
            "name": self.name,
            "lambda": self.lam,
            "n_features": self._n_features,
            "n_zero_coefs": n_zero,
            "n_nonzero_coefs": self._n_features - n_zero,
            "n_iterations": self.n_iter_,
        }


# ═══════════════════════════════════════════════════════════════
#  7. K-FOLD CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════

def kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_class: type,
    lambdas: np.ndarray,
    k: int = 5,
    seed: int = SEED,
    **model_kwargs,
) -> tuple[float, dict]:
    """
    K-fold Cross-Validation to select best hyperparameter λ.

    WHY k-fold instead of single validation split: Reduces variance of the
    performance estimate. The PDF (§2.5) explicitly requires k-fold CV.

    Args:
        X, y: training data only (do NOT pass test set)
        model_class: RidgeModel or LassoModel
        lambdas: array of λ values to evaluate
        k: number of folds (5 or 10)
        seed: for reproducible fold splits

    Returns:
        best_lambda: λ with lowest mean CV MSE
        cv_results: dict[float, dict] with mean_mse, std_mse, fold_scores
    """
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    # Split indices into k folds
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[:n % k] += 1  # distribute remainder
    folds = []
    start = 0
    for size in fold_sizes:
        folds.append(indices[start:start + size])
        start += size

    cv_results = {}

    for lam in lambdas:
        fold_mses = []
        for i in range(k):
            # Validation fold = i, training folds = rest
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model = model_class(lam=lam, **model_kwargs)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            mse = float(np.mean((y_val - y_pred) ** 2))
            fold_mses.append(mse)

        cv_results[float(lam)] = {
            "mean_mse": float(np.mean(fold_mses)),
            "std_mse": float(np.std(fold_mses)),
            "fold_scores": fold_mses,
        }

    # Best λ = lowest mean MSE
    best_lambda = min(cv_results, key=lambda l: cv_results[l]["mean_mse"])
    return best_lambda, cv_results


def plot_cv_curve(
    cv_results: dict,
    best_lambda: float,
    model_name: str = "Ridge",
    save: bool = True,
) -> None:
    """Plot CV MSE vs λ with error bars and highlight best λ."""
    lambdas = sorted(cv_results.keys())
    means = [cv_results[l]["mean_mse"] for l in lambdas]
    stds = [cv_results[l]["std_mse"] for l in lambdas]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(lambdas, means, yerr=stds, fmt="o-", capsize=3,
                color="#2196F3", ecolor="#90CAF9", label="Mean CV MSE ± 1 std")
    ax.axvline(best_lambda, color="#F44336", linestyle="--", linewidth=1.5,
               label=f"Best λ = {best_lambda:.4g}")
    ax.set_xscale("log")
    ax.set_xlabel("λ (regularization strength)", fontsize=12)
    ax.set_ylabel("Mean Squared Error (CV)", fontsize=12)
    ax.set_title(f"K-Fold Cross-Validation — {model_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fname = os.path.join(FIGURES_DIR, f"cv_curve_{model_name.lower()}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  → Saved: {fname}")
    plt.show()


# ═══════════════════════════════════════════════════════════════
#  8. RESIDUAL ANALYSIS (4 Diagnostic Plots)
# ═══════════════════════════════════════════════════════════════

def _lowess(x: np.ndarray, y: np.ndarray, frac: float = 0.3, n_points: int = 50) -> tuple:
    """
    Simple LOWESS (Locally Weighted Scatterplot Smoothing) from scratch.
    For each point, fit a weighted linear regression using a tri-cube kernel.
    """
    x_sorted = np.sort(x)
    x_grid = np.linspace(x_sorted[0], x_sorted[-1], n_points)
    y_smooth = np.zeros(n_points)
    h = int(np.ceil(frac * len(x)))

    for i, x0 in enumerate(x_grid):
        dists = np.abs(x - x0)
        idx = np.argsort(dists)[:h]
        x_local, y_local = x[idx], y[idx]
        max_dist = dists[idx[-1]] + 1e-10

        # Tri-cube kernel weights
        u = dists[idx] / max_dist
        w = (1 - u ** 3) ** 3

        # Weighted linear regression: minimize Σwᵢ(yᵢ - a - bxᵢ)²
        W = np.diag(w)
        X_local = np.column_stack([np.ones(len(x_local)), x_local])
        try:
            beta = np.linalg.solve(X_local.T @ W @ X_local, X_local.T @ W @ y_local)
            y_smooth[i] = beta[0] + beta[1] * x0
        except np.linalg.LinAlgError:
            y_smooth[i] = np.mean(y_local)

    return x_grid, y_smooth


def plot_residuals_vs_fitted(y_true, y_pred, save=True):
    """
    Plot 1/4: Residuals vs Fitted values.
    WHY: Checks linearity assumption and homoscedasticity.
    A random scatter around 0 indicates good model fit.
    """
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(y_pred, residuals, alpha=0.15, s=5, color="#1976D2", rasterized=True)
    ax.axhline(0, color="#F44336", linewidth=1.5, linestyle="--")

    # LOWESS smoother
    sample_idx = np.random.choice(len(y_pred), min(5000, len(y_pred)), replace=False)
    x_sm, y_sm = _lowess(y_pred[sample_idx], residuals[sample_idx])
    ax.plot(x_sm, y_sm, color="#FF9800", linewidth=2.5, label="LOWESS")

    ax.set_xlabel("Fitted values (ŷ)", fontsize=12)
    ax.set_ylabel("Residuals (y − ŷ)", fontsize=12)
    ax.set_title("Residuals vs Fitted", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fname = os.path.join(FIGURES_DIR, "residuals_vs_fitted.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  → Saved: {fname}")
    plt.show()


def plot_qq(y_true, y_pred, save=True):
    """
    Plot 2/4: Q-Q Plot of standardized residuals.
    WHY: Checks normality of residuals (Gauss-Markov assumption).
    Points following the 45° line indicate normally distributed residuals.
    """
    residuals = y_true - y_pred
    # Standardize residuals
    std_residuals = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-10)
    std_residuals_sorted = np.sort(std_residuals)

    n = len(std_residuals_sorted)
    # Theoretical quantiles from N(0,1)
    theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

    fig, ax = plt.subplots(figsize=(8, 8))
    # Subsample for large datasets
    if n > 10000:
        idx = np.linspace(0, n - 1, 10000, dtype=int)
        ax.scatter(theoretical[idx], std_residuals_sorted[idx],
                   alpha=0.3, s=5, color="#1976D2", rasterized=True)
    else:
        ax.scatter(theoretical, std_residuals_sorted,
                   alpha=0.3, s=5, color="#1976D2")

    # 45° reference line
    lim = max(abs(theoretical.min()), abs(theoretical.max()))
    ax.plot([-lim, lim], [-lim, lim], color="#F44336", linewidth=1.5, linestyle="--",
            label="y = x (ideal)")

    ax.set_xlabel("Theoretical Quantiles (N(0,1))", fontsize=12)
    ax.set_ylabel("Standardized Residuals", fontsize=12)
    ax.set_title("Normal Q-Q Plot", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fname = os.path.join(FIGURES_DIR, "qq_plot.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  → Saved: {fname}")
    plt.show()


def plot_scale_location(y_true, y_pred, save=True):
    """
    Plot 3/4: Scale-Location (Spread-Level) plot.
    WHY: Checks homoscedasticity — if the LOWESS line is flat,
    the variance of residuals is constant across fitted values.
    """
    residuals = y_true - y_pred
    std_residuals = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-10)
    sqrt_abs_std_res = np.sqrt(np.abs(std_residuals))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, sqrt_abs_std_res, alpha=0.15, s=5, color="#1976D2", rasterized=True)

    sample_idx = np.random.choice(len(y_pred), min(5000, len(y_pred)), replace=False)
    x_sm, y_sm = _lowess(y_pred[sample_idx], sqrt_abs_std_res[sample_idx])
    ax.plot(x_sm, y_sm, color="#FF9800", linewidth=2.5, label="LOWESS")

    ax.set_xlabel("Fitted values (ŷ)", fontsize=12)
    ax.set_ylabel("√|Standardized Residuals|", fontsize=12)
    ax.set_title("Scale-Location Plot", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        fname = os.path.join(FIGURES_DIR, "scale_location.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  → Saved: {fname}")
    plt.show()


def plot_cooks_distance(X, y, model, n_display=10000, save=True):
    """
    Plot 4/4: Cook's Distance.
    Dᵢ = (eᵢ²) / (p × MSE) × hᵢᵢ / (1 − hᵢᵢ)²
    WHY: Identifies influential observations that disproportionately affect β̂.

    Uses diagonal-only hat matrix computation for memory efficiency on large data.
    Subsamples for display readability.
    """
    X_b = _add_intercept(X)
    n, p_full = X_b.shape
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = np.mean(residuals ** 2)

    # Efficient diagonal of hat matrix: hᵢᵢ = xᵢᵀ(XᵀX)⁻¹xᵢ
    XtX_inv = np.linalg.inv(X_b.T @ X_b)
    h_diag = np.sum((X_b @ XtX_inv) * X_b, axis=1)

    # Cook's Distance
    cooks_d = (residuals ** 2) / (p_full * mse) * h_diag / ((1 - h_diag) ** 2 + 1e-10)

    # Subsample for readability
    if n > n_display:
        idx = np.random.choice(n, n_display, replace=False)
        idx = np.sort(idx)
        cooks_d_plot = cooks_d[idx]
    else:
        idx = np.arange(n)
        cooks_d_plot = cooks_d

    threshold = 4.0 / n

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(cooks_d_plot)), cooks_d_plot, width=1.0, color="#1976D2", alpha=0.7)
    ax.axhline(threshold, color="#F44336", linestyle="--", linewidth=1.5,
               label=f"Threshold = 4/n = {threshold:.2e}")
    ax.set_xlabel("Observation Index", fontsize=12)
    ax.set_ylabel("Cook's Distance", fontsize=12)
    ax.set_title("Cook's Distance — Influential Observations", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save:
        fname = os.path.join(FIGURES_DIR, "cooks_distance.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  → Saved: {fname}")
    plt.show()

    n_influential = int(np.sum(cooks_d > threshold))
    print(f"  → {n_influential} influential observations (Cook's D > 4/n)")
    return cooks_d


def run_residual_analysis(X, y, model, save=True):
    """Run all 4 diagnostic plots for a given model."""
    y_pred = model.predict(X)
    print(f"\n{'='*60}")
    print(f"  RESIDUAL ANALYSIS — {model.name}")
    print(f"{'='*60}")
    plot_residuals_vs_fitted(y, y_pred, save=save)
    plot_qq(y, y_pred, save=save)
    plot_scale_location(y, y_pred, save=save)
    plot_cooks_distance(X, y, model, save=save)


# ═══════════════════════════════════════════════════════════════
#  9. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════

def plot_feature_importance(feature_names, model, top_n=20, save=True):
    """
    Plot standardized regression coefficients as horizontal bar chart.
    Since input data is already standardized (std≈1), the raw coefficients
    approximate the standardized coefficients. Sort by absolute magnitude.
    Color: blue for positive, red for negative.
    """
    coefs = model.get_coefficients()
    if len(feature_names) != len(coefs):
        feature_names = [f"Feature_{i}" for i in range(len(coefs))]

    # Sort by absolute value
    sorted_idx = np.argsort(np.abs(coefs))[::-1][:top_n]
    sorted_coefs = coefs[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    colors = ["#1976D2" if c >= 0 else "#D32F2F" for c in sorted_coefs]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    bars = ax.barh(range(len(sorted_coefs)), sorted_coefs, color=colors, alpha=0.85)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient Value (standardized)", fontsize=12)
    ax.set_title(f"Feature Importance — {model.name} (Top {top_n})", fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    if save:
        fname = os.path.join(FIGURES_DIR, "feature_importance.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  → Saved: {fname}")
    plt.show()


# ═══════════════════════════════════════════════════════════════
#  10. MODEL COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════

def compare_models(
    models: dict[str, BaseModel],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Generate comparison table for all trained models.
    Accepts ANY model with BaseModel interface (including future Kernel/Bayesian).
    Sorted by RMSE ascending (best first).
    """
    rows = []
    for name, model in models.items():
        metrics = model.evaluate(X_test, y_test)
        smry = model.summary()
        n_feat = smry.get("n_features_selected", smry.get("n_features", "?"))
        rows.append({
            "Model": name,
            "MAE": round(metrics["MAE"], 4),
            "RMSE": round(metrics["RMSE"], 4),
            "R²": round(metrics["R2"], 4),
            "Adj_R²": round(metrics["Adj_R2"], 4),
            "Num_Features": n_feat,
        })

    df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

    # Print formatted table
    print(f"\n{'='*70}")
    print(f"{'MODEL COMPARISON — PM2.5':^70}")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print(f"{'='*70}")
    print(f"  Best model: {df.iloc[0]['Model']} (RMSE = {df.iloc[0]['RMSE']:.4f})")

    # Bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    model_names = df["Model"].tolist()
    colors = sns.color_palette("viridis", len(model_names))

    for ax, metric in zip(axes, ["MAE", "RMSE", "R²"]):
        values = df[metric].tolist()
        bars = ax.bar(model_names, values, color=colors, alpha=0.85)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_ylabel(metric, fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Model Comparison — Beijing PM2.5", fontsize=16, fontweight="bold")
    plt.tight_layout()

    fname = os.path.join(FIGURES_DIR, "model_comparison.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  → Saved: {fname}")
    plt.show()

    return df


# ═══════════════════════════════════════════════════════════════
#  11. MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

def run_model_comparison(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    k_folds: int = 5,
    save_figures: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """
    Complete model training and comparison pipeline.

    Steps:
        1. Train OLS Basic on all features
        2. Train OLS with Backward Elimination (feature selection)
        3. Run k-fold CV to find best λ for Ridge (plot CV curve)
        4. Train Ridge with best λ
        5. Run k-fold CV to find best λ for Lasso (plot CV curve)
        6. Train Lasso with best λ
        7. Evaluate all models on test set → comparison table
        8. Select best model → residual analysis (4 plots)
        9. Feature importance plot for best model
       10. Print summary and conclusions

    Returns:
        results: dict of {model_name: {model, metrics, predictions, ...}}
        comparison_df: DataFrame with all metrics
    """
    results = {}
    lambdas = np.logspace(-4, 4, 50)

    # ── Step 1: OLS Basic ────────────────────────────────────
    print("\n[1/6] Training OLS Basic...")
    ols = OLSBasicModel()
    ols.fit(X_train, y_train)
    ols_inf = ols.inference(X_train, y_train)
    results["OLS Basic"] = {
        "model": ols,
        "inference": ols_inf,
    }
    print(f"  ✓ OLS Basic fitted. R² (train) = {compute_r2(y_train, ols.predict(X_train)):.4f}")

    # ── Step 2: OLS Feature Selected ─────────────────────────
    print("\n[2/6] Training OLS with Backward Elimination...")
    ols_sel = OLSFeatureSelectedModel(significance_level=0.05, vif_threshold=10.0)
    ols_sel.fit(X_train, y_train)
    sel_summary = ols_sel.summary()
    sel_names = [feature_names[i] for i in ols_sel.selected_features]
    results["OLS Feature Selected"] = {
        "model": ols_sel,
        "selected_features": sel_names,
        "elimination_history": ols_sel.elimination_history,
    }
    print(f"  ✓ Features: {sel_summary['n_features_original']} → {sel_summary['n_features_selected']}")
    print(f"  ✓ Eliminated {len(ols_sel.elimination_history)} features")

    # ── Step 3 & 4: Ridge with CV ────────────────────────────
    print(f"\n[3/6] Cross-validating Ridge (k={k_folds})...")
    best_lam_ridge, cv_ridge = kfold_cv(X_train, y_train, RidgeModel, lambdas, k=k_folds)
    plot_cv_curve(cv_ridge, best_lam_ridge, "Ridge", save=save_figures)

    ridge = RidgeModel(lam=best_lam_ridge)
    ridge.fit(X_train, y_train)
    ridge_name = f"Ridge (λ={best_lam_ridge:.4g})"
    results[ridge_name] = {
        "model": ridge,
        "best_lambda": best_lam_ridge,
        "cv_results": cv_ridge,
    }
    print(f"  ✓ Best λ = {best_lam_ridge:.4g}")

    # ── Step 5 & 6: Lasso with CV ────────────────────────────
    print(f"\n[4/6] Cross-validating Lasso (k={k_folds})...")
    best_lam_lasso, cv_lasso = kfold_cv(
        X_train, y_train, LassoModel, lambdas, k=k_folds, max_iter=2000
    )
    plot_cv_curve(cv_lasso, best_lam_lasso, "Lasso", save=save_figures)

    lasso = LassoModel(lam=best_lam_lasso, max_iter=2000)
    lasso.fit(X_train, y_train)
    lasso_name = f"Lasso (λ={best_lam_lasso:.4g})"
    zero_coefs = lasso.get_zero_coefficients(feature_names)
    results[lasso_name] = {
        "model": lasso,
        "best_lambda": best_lam_lasso,
        "cv_results": cv_lasso,
        "zero_coefficients": zero_coefs,
    }
    print(f"  ✓ Best λ = {best_lam_lasso:.4g}")
    print(f"  ✓ {len(zero_coefs)} features zeroed out: {zero_coefs[:5]}{'...' if len(zero_coefs) > 5 else ''}")

    # ── Step 7: Comparison table ─────────────────────────────
    print("\n[5/6] Comparing models on test set...")
    models_dict = {name: info["model"] for name, info in results.items()}
    comparison_df = compare_models(models_dict, X_test, y_test)

    # ── Step 8 & 9: Best model analysis ──────────────────────
    best_name = comparison_df.iloc[0]["Model"]
    best_model = results[best_name]["model"]
    print(f"\n[6/6] Analyzing best model: {best_name}")

    run_residual_analysis(X_test, y_test, best_model, save=save_figures)
    plot_feature_importance(feature_names, best_model, save=save_figures)

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Best model       : {best_name}")
    print(f"  Test RMSE        : {comparison_df.iloc[0]['RMSE']:.4f}")
    print(f"  Test R²          : {comparison_df.iloc[0]['R²']:.4f}")
    print(f"  Models compared  : {len(models_dict)}")
    print(f"  Figures saved to : {FIGURES_DIR}")
    print(f"{'='*60}")

    return results, comparison_df


# ═══════════════════════════════════════════════════════════════
#  12. UNIT TESTS
# ═══════════════════════════════════════════════════════════════

def _run_tests():
    """Unit tests for all core functions. ≥ 2 tests per function."""
    print("\n" + "=" * 60)
    print("  RUNNING UNIT TESTS")
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

    # ── Metrics tests ────────────────────────────────────────
    y_t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_p = np.array([1.1, 2.0, 2.9, 4.1, 5.0])

    check("MAE: basic", abs(compute_mae(y_t, y_p) - 0.06) < 0.01)
    check("MAE: perfect", compute_mae(y_t, y_t) == 0.0)

    check("RMSE: basic", abs(compute_rmse(y_t, y_p) - np.sqrt(0.03 / 5)) < 0.01)
    check("RMSE: perfect", compute_rmse(y_t, y_t) == 0.0)

    check("R²: perfect fit", compute_r2(y_t, y_t) == 1.0)
    check("R²: reasonable", 0.99 < compute_r2(y_t, y_p) < 1.0)

    check("Adj R²: perfect", compute_adjusted_r2(y_t, y_t, 1) == 1.0)
    check("Adj R²: ≤ R²", compute_adjusted_r2(y_t, y_p, 2) <= compute_r2(y_t, y_p))

    # ── OLS Basic tests ──────────────────────────────────────
    # Test: y = 2x₁ - x₂ + 3 (known solution)
    np.random.seed(SEED)
    n = 200
    X_syn = np.random.randn(n, 2)
    y_syn = 2.0 * X_syn[:, 0] - 1.0 * X_syn[:, 1] + 3.0 + np.random.randn(n) * 0.01

    ols_test = OLSBasicModel()
    ols_test.fit(X_syn, y_syn)

    check("OLS: intercept ≈ 3", abs(ols_test.get_intercept() - 3.0) < 0.1)
    check("OLS: coef[0] ≈ 2", abs(ols_test.get_coefficients()[0] - 2.0) < 0.1)
    check("OLS: coef[1] ≈ -1", abs(ols_test.get_coefficients()[1] + 1.0) < 0.1)

    # Compare with np.linalg.lstsq
    X_b = _add_intercept(X_syn)
    beta_lstsq, _, _, _ = np.linalg.lstsq(X_b, y_syn, rcond=None)
    check("OLS: matches lstsq", np.allclose(ols_test.beta_hat, beta_lstsq, atol=1e-8))

    # OLS inference
    inf = ols_test.inference(X_syn, y_syn)
    check("OLS inference: p-values shape", len(inf["p_values"]) == 3)
    check("OLS inference: all p < 0.05", np.all(inf["p_values"] < 0.05))

    # ── Ridge tests ──────────────────────────────────────────
    ridge_0 = RidgeModel(lam=0.0)
    ridge_0.fit(X_syn, y_syn)
    check("Ridge(λ=0) ≈ OLS", np.allclose(ridge_0.beta_hat, ols_test.beta_hat, atol=1e-6))

    ridge_big = RidgeModel(lam=1e6)
    ridge_big.fit(X_syn, y_syn)
    check("Ridge(λ→∞): coefs → 0", np.all(np.abs(ridge_big.get_coefficients()) < 0.01))

    # ── Lasso tests ──────────────────────────────────────────
    lasso_0 = LassoModel(lam=0.0, max_iter=5000, tol=1e-6)
    lasso_0.fit(X_syn, y_syn)
    check("Lasso(λ=0) ≈ OLS", np.allclose(
        lasso_0.get_coefficients(), ols_test.get_coefficients(), atol=0.05
    ))

    lasso_big = LassoModel(lam=10.0)
    lasso_big.fit(X_syn, y_syn)
    n_zero = sum(abs(c) < 1e-10 for c in lasso_big.get_coefficients())
    check("Lasso(λ=10): sparsity", n_zero >= 1)

    # ── VIF tests ────────────────────────────────────────────
    X_indep = np.random.randn(100, 3)
    vifs = compute_vif(X_indep)
    check("VIF: independent features ≈ 1", np.all(vifs < 2.0))

    X_collinear = np.random.randn(100, 2)
    X_collinear = np.column_stack([X_collinear, X_collinear[:, 0] * 0.99 + 0.01 * np.random.randn(100)])
    vifs_c = compute_vif(X_collinear)
    check("VIF: collinear feature > 10", np.max(vifs_c) > 10)

    # ── K-fold CV tests ──────────────────────────────────────
    lambdas_test = np.array([0.001, 0.01, 0.1, 1.0, 10.0])
    best_lam, cv_res = kfold_cv(X_syn, y_syn, RidgeModel, lambdas_test, k=3)
    check("CV: returns valid lambda", best_lam in lambdas_test)
    check("CV: best λ is small (data fits well)", best_lam < 1.0)

    # ── Summary ──────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    success = _run_tests()
    if not success:
        exit(1)


