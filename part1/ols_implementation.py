import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def ols_fit(X, y):
    """
    Tính beta_hat = (X^T X)^-1 X^T y và sigma2_hat = RSS / (n - p - 1)
    """
    n, p_plus_1 = X.shape
    # Tính beta_hat bằng normal equations: (X^T X) beta = X^T y
    beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
    
    # Tính RSS và sigma2_hat
    y_hat = X @ beta_hat
    rss = np.sum((y - y_hat) ** 2)
    sigma2_hat = rss / (n - p_plus_1)
    
    return beta_hat, sigma2_hat


def hat_matrix(X):
    """
    Tính ma trận chiếu H = X(X^T X)^-1 X^T
    """
    H = X @ np.linalg.solve(X.T @ X, X.T)
    assert np.allclose(H @ H, H, atol=1e-9), "FAIL: H^2 ≠ H"
    assert np.allclose(H, H.T, atol=1e-9),   "FAIL: H không đối xứng"
    rank = np.linalg.matrix_rank(H)
    assert rank == X.shape[1], f"FAIL: rank(H)={rank} ≠ {X.shape[1]}"
    return H


def model_metrics(y, y_hat, p):
    """
    Tính RSS, TSS, R^2, R^2_adj, F_stat
    p: số lượng biến (không tính intercept)
    """
    n = len(y)
    rss = np.sum((y - y_hat) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    
    r2 = 1 - (rss / tss)
    r2_adj = 1 - ((n - 1) / (n - p - 1)) * (1 - r2)
    
    # F-statistic: ((TSS - RSS) / p) / (RSS / (n - p - 1))
    f_stat = ((tss - rss) / p) / (rss / (n - p - 1))

    f_pvalue = 1 - stats.f.cdf(f_stat, dfn=p, dfd=n - p - 1)
    
    return rss, tss, r2, r2_adj, f_stat, f_pvalue


def coef_inference(X, y, beta_hat, sigma2_hat):
    """
    Tính se, t_stat, p_value, khoảng tin cậy 95%
    """
    n, p_plus_1 = X.shape
    # Ma trận hiệp phương sai của beta: sigma2 * (X^T X)^-1
    var_beta = sigma2_hat * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diagonal(var_beta))
    
    t_stat = beta_hat / se
    
    # Bậc tự do: n - (p + 1)
    df = n - p_plus_1
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
    
    # Khoảng tin cậy 95% (alpha = 0.05)
    t_critical = stats.t.ppf(0.975, df)
    ci_lower = beta_hat - t_critical * se
    ci_upper = beta_hat + t_critical * se
    
    return se, t_stat, p_value, (ci_lower, ci_upper)


def vif(X):
    """
    Tính VIF (Variance Inflation Factor) cho từng biến trong X.
    Giả sử cột đầu của X là intercept (toàn 1) → bỏ qua.
    
    Tham số:
        X: np.ndarray, shape (n, p+1), ma trận thiết kế có intercept
        
    Trả về:
        vif_values: danh sách các giá trị VIF cho từng biến (trừ intercept)
    """
    n, p = X.shape
    has_intercept = np.allclose(X[:, 0], 1)
    start_idx = 1 if has_intercept else 0

    vif_values = []
    for j in range(start_idx, p):
        X_j = X[:, j]
        cols = [i for i in range(p) if i != j]
        X_other = X[:, cols]

        # OLS: hồi quy X_j theo các biến còn lại
        beta_hat = np.linalg.solve(X_other.T @ X_other, X_other.T @ X_j)
        X_j_hat = X_other @ beta_hat

        rss = np.sum((X_j - X_j_hat) ** 2)
        tss = np.sum((X_j - np.mean(X_j)) ** 2)

        if tss < 1e-10:
            vif_values.append(np.inf)
        else:
            r2 = 1 - rss / tss
            r2 = min(r2, 1 - 1e-10)
            vif_values.append(1 / (1 - r2))

    # In kết quả
    print("=== Kết quả VIF ===")
    for j, v in enumerate(vif_values):
        flag = " -> Đa cộng tuyến!" if v > 10 else ""
        print(f"  VIF(X{j + start_idx}) = {v:.4f}{flag}")

    return vif_values


def run_monte_carlo(n_simulations=1000, n_samples=100, n_features=3, seed=42):
    """
    Mô phỏng Monte Carlo để chứng minh định lý Gauss-Markov:
    1. Kỳ vọng E[beta_hat] = beta (Tính không chệch - Unbiasedness).
    2. Phương sai nhỏ nhất (Tính hiệu quả - Efficiency / BLUE).
    """
    np.random.seed(seed)
    
    # 1. Khởi tạo X cố định và beta thực
    X_base = np.random.randn(n_samples, n_features)
    X = np.hstack([np.ones((n_samples, 1)), X_base]) # Thêm cột intercept
    p = n_features + 1
    
    true_beta = np.array([2.5, 1.5, -2.0, 3.0])
    sigma = 1.5
    
    # 2. Xây dựng một ước lượng tuyến tính không chệch khác: beta_alt = (W) y
    # Để không chệch, W @ X phải bằng đơn vị I.
    # W_ols = (X^T X)^{-1} X^T
    # W_alt = W_ols + C, với C @ X = 0
    
    # Ma trận Hat matrix và ma trận dư M
    H = X @ np.linalg.solve(X.T @ X, X.T)
    M = np.eye(n_samples) - H # Chiếu lên không gian trực giao của X
    
    # Tạo ma trận C ngẫu nhiên thỏa mãn C @ X = 0
    random_mat = np.random.randn(p, n_samples)
    C = random_mat @ M
    
    # Kiểm tra điều kiện C @ X = 0
    assert np.allclose(C @ X, 0), "FAIL: C @ X không bằng 0, estimator sẽ bị chệch!"
    
    W_ols = np.linalg.solve(X.T @ X, X.T)
    W_alt = W_ols + C
    
    beta_hat_ols_list = []
    beta_hat_alt_list = []
    
    # 3. Chạy mô phỏng Monte Carlo
    print(f"Đang chạy {n_simulations} lần mô phỏng...")
    for i in range(n_simulations):
        # Sinh nhiễu trắng chuẩn (Gauss)
        epsilon = np.random.normal(0, sigma, n_samples)
        
        # Tạo dữ liệu quan sát y
        y = X @ true_beta + epsilon
        
        # Ước lượng OLS
        beta_ols = W_ols @ y
        beta_hat_ols_list.append(beta_ols)
        
        # Ước lượng thay thế (cũng tuyến tính và không chệch)
        beta_alt = W_alt @ y
        beta_hat_alt_list.append(beta_alt)
        
    beta_hat_ols_list = np.array(beta_hat_ols_list)
    beta_hat_alt_list = np.array(beta_hat_alt_list)
    
    # 4. Tính toán thống kê
    mean_ols = np.mean(beta_hat_ols_list, axis=0)
    mean_alt = np.mean(beta_hat_alt_list, axis=0)
    
    var_ols = np.var(beta_hat_ols_list, axis=0)
    var_alt = np.var(beta_hat_alt_list, axis=0)
    
    # 5. Hiển thị kết quả
    print("="*60)
    print("KẾT QUẢ MÔ PHỎNG MONTE CARLO (KIỂM CHỨNG GAUSS-MARKOV)")
    print("="*60)
    print(f"Số mẫu: {n_samples}, Số đặc trưng: {n_features}")
    print(f"Beta thực tế: {true_beta}")
    print("-" * 60)
    
    print("1. KIỂM CHỨNG TÍNH KHÔNG CHỆCH (E[beta_hat] = beta):")
    print(f"   Trung bình Beta OLS: {mean_ols.round(4)}")
    print(f"   Trung bình Beta Alt: {mean_alt.round(4)}")
    print("   => Nhận xét: Cả hai đều hội tụ về giá trị thực.")
    print("-" * 60)
    
    print("2. KIỂM CHỨNG PHƯƠNG SAI NHỎ NHẤT (BLUE):")
    print(f"   Phương sai Beta OLS: {var_ols.round(4)}")
    print(f"   Phương sai Beta Alt: {var_alt.round(4)}")
    print("   => Nhận xét: Phương sai của OLS nhỏ hơn đáng kể.")
    
    # 6. Trực quan hóa kết quả cho hệ số beta_1
    plt.figure(figsize=(10, 6))
    plt.hist(beta_hat_alt_list[:, 1], bins=50, alpha=0.4, label='Ước lượng thay thế (Alternative)', color='orange')
    plt.hist(beta_hat_ols_list[:, 1], bins=50, alpha=0.6, label='Ước lượng OLS', color='blue')
    
    plt.axvline(true_beta[1], color='red', linestyle='--', linewidth=2, label='Beta_1 thực tế')
    
    plt.title('So sánh phân phối của ước lượng OLS và một ước lượng không chệch khác')
    plt.xlabel('Giá trị beta_1 ước lượng được')
    plt.ylabel('Tần suất (Số lần)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # Chạy mô phỏng Monte Carlo để kiểm chứng định lý Gauss-Markov
    run_monte_carlo()
