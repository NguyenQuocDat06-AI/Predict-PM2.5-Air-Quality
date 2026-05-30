import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ols_implementation import manual_transpose, manual_matmul, manual_solve, manual_inv

def get_residuals(X, y, beta_hat):
    """
    Tính vector phần dư: e = y - X * beta_hat

    Parameters
    ----------
    X        : list[list[float]]  — ma trận thiết kế (đã gồm cột bias)
    y        : list[float]        — vector phản hồi quan sát
    beta_hat : list[float]        — vector hệ số OLS ước lượng

    Returns
    -------
    residuals : list[float]       — y_hat và vector phần dư e_i = y_i - y_hat_i
    y_hat     : list[float]       — giá trị dự đoán
    """
    y_list = y.tolist() if hasattr(y, "tolist") else y
    X_list = X.tolist() if hasattr(X, "tolist") else X
    n = len(y_list)
    y_hat = manual_matmul(X_list, beta_hat)
    residuals = [y_list[i] - y_hat[i] for i in range(n)]
    return residuals, y_hat


def get_standardized_residuals(X, residuals, sigma2_hat):
    """
    Tính standardized residuals và leverage (h_ii).

    Parameters
    ----------
    X          : list[list[float]]  — ma trận thiết kế (đã gồm cột bias)
    residuals  : list[float]        — vector phần dư thô
    sigma2_hat : float              — ước lượng phương sai nhiễu

    Returns
    -------
    std_res : list[float]   — standardized residuals
    h       : list[float]   — leverage h_ii (đường chéo hat matrix)
    """
    X_list = X.tolist() if hasattr(X, "tolist") else X
    n = len(X_list)
    p = len(X_list[0])  # số tham số (gồm intercept)

    Xt = manual_transpose(X_list)
    XtX_inv = manual_inv(manual_matmul(Xt, X_list))

    h = []
    for i in range(n):
        xi = X_list[i]
        # h_ii = xi^T * (X^T X)^{-1} * xi
        tmp = manual_matmul([xi], XtX_inv)[0]
        hii = sum(tmp[k] * xi[k] for k in range(p))
        h.append(hii)

    std_res = []
    for i in range(n):
        denom = math.sqrt(max(sigma2_hat * (1 - h[i]), 1e-12))
        std_res.append(residuals[i] / denom)

    return std_res, h

def cook_distance(std_res, h, p):
    """
    Tính Cook's Distance cho từng quan sát.

    Công thức: D_i = (e_i_std^2 * h_ii) / (p * (1 - h_ii))
    Tương đương viết lại: D_i = (e_i_std^2 / p) * (h_ii / (1 - h_ii))

    Parameters
    ----------
    std_res : list[float]  — standardized residuals (từ get_standardized_residuals)
    h       : list[float]  — leverage h_ii (từ get_standardized_residuals)
    p       : int          — số tham số trong mô hình, BAO GỒM intercept
                             (= số cột của design matrix X, tức len(X[0]))

    Returns
    -------
    d : list[float]  — Cook's Distance D_i cho mỗi quan sát
    """
    d = []
    for i in range(len(std_res)):
        # D_i = (std_res_i^2 * h_ii) / (p * (1 - h_ii))
        val = (std_res[i] ** 2 * h[i]) / (p * (1 - h[i]))
        d.append(val)
    return d


def residual_plots(X, y, beta_hat):
    """
    Vẽ 4 biểu đồ phân tích phần dư:
      1. Residuals vs Fitted  — kiểm tra tính tuyến tính và phương sai đồng đều
      2. Normal Q-Q Plot      — kiểm tra phân phối chuẩn của phần dư
      3. Scale-Location       — kiểm tra homoscedasticity
      4. Cook's Distance      — phát hiện điểm ảnh hưởng

    Parameters
    ----------
    X        : list[list[float]]  — ma trận thiết kế (đã gồm cột bias)
    y        : list[float]        — vector phản hồi quan sát
    beta_hat : list[float]        — vector hệ số OLS ước lượng

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    X_list = X.tolist() if hasattr(X, "tolist") else list(X)
    y_list = y.tolist() if hasattr(y, "tolist") else list(y)
    n = len(y_list)
    p = len(X_list[0])  # số tham số (gồm intercept/bias column)

    # --- Tính residuals và y_hat (tái sử dụng get_residuals) ---
    residuals, y_hat = get_residuals(X_list, y_list, beta_hat)

    # --- Tính sigma2_hat từ residuals ---
    rss = sum(r ** 2 for r in residuals)
    sigma2_hat = rss / max(n - p, 1)

    # --- Tính Standardized residuals, leverage, Cook's D ---
    std_res, h = get_standardized_residuals(X_list, residuals, sigma2_hat)
    cook_d = cook_distance(std_res, h, p)

    # --- Q-Q plot — quantile lý thuyết chuẩn ---
    def norm_ppf_approx(prob):
        """
        Xấp xỉ phân vị chuẩn tắc (inverse CDF) bằng thuật toán Beasley-Springer-Moro.
        """
        a = [2.515517, 0.802853, 0.010328]
        b = [1.432788, 0.189269, 0.001308]
        if prob <= 0 or prob >= 1:
            return float('nan')
        if prob < 0.5:
            t = math.sqrt(-2 * math.log(prob))
            sign = -1
        else:
            t = math.sqrt(-2 * math.log(1 - prob))
            sign = 1
        num = a[0] + a[1] * t + a[2] * t ** 2
        den = 1 + b[0] * t + b[1] * t ** 2 + b[2] * t ** 3
        return sign * (t - num / den)

    sorted_res = sorted(std_res)
    # Dùng plotting position chuẩn của R: (i - 3/8) / (n + 1/4)
    # (thay vì Hazen: (i + 0.5) / n) để so sánh tốt hơn với statsmodels/R
    theoretical_q = [
        norm_ppf_approx((i + 1 - 3 / 8) / (n + 1 / 4)) for i in range(n)
    ]

    # sqrt(|standardized residuals|) cho Scale-Location
    sqrt_abs_std = [math.sqrt(abs(r)) for r in std_res]

    # Ngưỡng Cook's Distance phổ biến nhất
    threshold = 4 / n

    # --- Vẽ ---
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Residual Diagnostic Plots", fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # 1. Residuals vs Fitted
    ax1.scatter(y_hat, residuals, color="#4C72B0", alpha=0.7, edgecolors="white", s=50)
    ax1.axhline(0, color="red", linestyle="--", linewidth=1)
    ax1.set_xlabel("Fitted values  $\\hat{y}$")
    ax1.set_ylabel("Residuals")
    ax1.set_title("1. Residuals vs Fitted")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # 2. Normal Q-Q
    ax2.scatter(theoretical_q, sorted_res, color="#DD8452", alpha=0.7, edgecolors="white", s=50)
    ax2.plot(theoretical_q, theoretical_q, color="red", linestyle="--", linewidth=1, label="y = x")
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Standardized Residuals")
    ax2.set_title("2. Normal Q-Q Plot")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # 3. Scale-Location (sqrt|std_res| vs Fitted)
    ax3.scatter(y_hat, sqrt_abs_std, color="#55A868", alpha=0.7, edgecolors="white", s=50)
    ax3.axhline(
        sum(sqrt_abs_std) / n, color="red", linestyle="--", linewidth=1, label="Mean"
    )
    ax3.set_xlabel("Fitted values  $\\hat{y}$")
    ax3.set_ylabel("$\\sqrt{|\\text{Standardized Residuals}|}$")
    ax3.set_title("3. Scale-Location")
    ax3.legend(fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.4)

    # 4. Cook's Distance (bar chart + annotation influential points)
    ax4.bar(range(n), cook_d, color="#C44E52", alpha=0.75, edgecolor="white")
    ax4.axhline(threshold, color="red", linestyle="--", linewidth=1,
                label=f"Threshold 4/n = {threshold:.3f}")
    # Ghi index của các điểm có ảnh hưởng lớn (Cook's D > 4/n)
    for idx, d_val in enumerate(cook_d):
        if d_val > threshold:
            ax4.annotate(str(idx), (idx, d_val),
                         textcoords="offset points", xytext=(0, 5), fontsize=7,
                         ha="center", color="darkred")
    ax4.set_xlabel("Observation index")
    ax4.set_ylabel("Cook's Distance")
    ax4.set_title("4. Cook's Distance")
    ax4.legend(fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.4)

    return fig