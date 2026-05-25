import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def residual_plots(X, y, beta_hat):
    """
    Vẽ 4 biểu đồ phân tích phần dư.
    """
    y_hat = X @ beta_hat
    residuals = y - y_hat
    n, p = X.shape  # p ở đây = p+1 (số tham số, gồm cả intercept)

    # Hat matrix
    H = X @ np.linalg.solve(X.T @ X, X.T)  # = X(X^TX)^{-1}X^T
    h_ii = np.diag(H)

    # Ước lượng sigma^2 không chệch
    sigma2_hat = np.sum(residuals ** 2) / (n - p)
    sigma_hat = np.sqrt(sigma2_hat)

    # Phần dư chuẩn hóa (Standardized residuals)
    denom = sigma_hat * np.sqrt(np.maximum(1 - h_ii, 1e-8))
    std_residuals = residuals / denom

    # ✅ Cook's Distance — công thức chuẩn
    # D_i = (e_i^2 / (p * sigma^2)) * (h_ii / (1 - h_ii)^2)
    cooks_d = (residuals ** 2 * h_ii) / \
              (p * sigma2_hat * (1 - h_ii) ** 2)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Residuals vs Fitted (Phần dư vs Giá trị dự báo)
    axs[0, 0].scatter(y_hat, residuals, alpha=0.5, edgecolors='k')
    axs[0, 0].axhline(0, color='r', linestyle='--')
    axs[0, 0].set_title('Residuals vs Fitted (Phần dư & Dự báo)')
    axs[0, 0].set_xlabel('Giá trị dự báo ($\\hat{y}$)')
    axs[0, 0].set_ylabel('Phần dư ($\\hat{\\varepsilon}$)')
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Normal Q-Q
    stats.probplot(std_residuals, dist="norm", plot=axs[0, 1])
    axs[0, 1].set_title('Normal Q-Q (Phân phối chuẩn)')
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Scale-Location
    axs[1, 0].scatter(y_hat, np.sqrt(np.abs(std_residuals)),
                      alpha=0.5, edgecolors='k')
    axs[1, 0].set_title('Scale-Location (Độ phân tán)')
    axs[1, 0].set_xlabel('Giá trị dự báo ($\\hat{y}$)')
    axs[1, 0].set_ylabel('$\\sqrt{|Phần\\ dư\\ chuẩn\\ hóa|}$')
    axs[1, 0].grid(True, alpha=0.3)

    # 4. Cook's Distance
    threshold = 4 / n
    axs[1, 1].stem(range(n), cooks_d, markerfmt=",", basefmt=" ")
    axs[1, 1].axhline(threshold, color='r', linestyle='--',
                      label=f'Ngưỡng 4/n = {threshold:.3f}')

    influential = np.where(cooks_d > threshold)[0]
    if len(influential) > 0:
        axs[1, 1].scatter(influential, cooks_d[influential],
                          color='red', zorder=5,
                          label=f'{len(influential)} điểm ảnh hưởng')

    axs[1, 1].set_title("Khoảng cách Cook (Cook's Distance)")
    axs[1, 1].set_xlabel('Chỉ số quan sát')
    axs[1, 1].set_ylabel("Khoảng cách Cook")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return std_residuals, cooks_d

if __name__ == "__main__":
    # Demo phân tích phần dư với dữ liệu giả lập
    np.random.seed(42)
    n, p = 100, 3
    X_base = np.random.randn(n, p)
    X = np.hstack([np.ones((n, 1)), X_base])
    true_beta = np.array([2, 1.5, -1, 0.5])
    y = X @ true_beta + np.random.normal(0, 1, n)
    
    # Ước lượng OLS (dùng công thức thủ công)
    beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
    
    # Vẽ biểu đồ
    residual_plots(X, y, beta_hat)