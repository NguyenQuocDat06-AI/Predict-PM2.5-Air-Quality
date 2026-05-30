import math
from ols_implementation import manual_transpose, manual_matmul, manual_solve, manual_inv
import matplotlib.pyplot as plt
import scipy.stats as stats
def get_residuals(X, y, beta_hat):
    y_list = y.tolist() if hasattr(y, "tolist") else y
    X_list = X.tolist() if hasattr(X, "tolist") else X
    n = len(y_list)
    y_hat = manual_matmul(X_list, beta_hat)
    return [y_list[i] - y_hat[i] for i in range(n)]

def get_standardized_residuals(X, residuals, sigma2_hat):
    X_list = X.tolist() if hasattr(X, "tolist") else X
    n = len(X_list)
    p = len(X_list[0])
    
    Xt = manual_transpose(X_list)
    XtX_inv = manual_inv(manual_matmul(Xt, X_list))
    
    h = []
    for i in range(n):
        xi = X_list[i]
        # h_ii = xi^T * XtX_inv * xi
        tmp = manual_matmul([xi], XtX_inv)[0]
        hii = sum(tmp[k] * xi[k] for k in range(p))
        h.append(hii)
        
    std_res = []
    for i in range(n):
        denom = math.sqrt(max(sigma2_hat * (1 - h[i]), 1e-12))
        std_res.append(residuals[i] / denom)
        
    return std_res, h

def cook_distance(std_res, h, p):
    d = []
    for i in range(len(std_res)):
        val = (std_res[i]**2 / p) * (h[i] / (1 - h[i]))
        d.append(val)
    return d

def residual_plots(X, y, beta_hat):
    """Vẽ 4 biểu đồ phân tích phần dư để kiểm tra các giả thiết OLS"""
    X_list = X.tolist() if hasattr(X, "tolist") else X
    y_list = y.tolist() if hasattr(y, "tolist") else y
    
    n = len(X_list)
    p_plus_1 = len(X_list[0])
    p = p_plus_1 - 1 
    
    residuals = get_residuals(X_list, y_list, beta_hat)
    
    rss = sum(r**2 for r in residuals)
    sigma2_hat = rss / (n - p_plus_1)
    
    std_residuals, h_ii = get_standardized_residuals(X_list, residuals, sigma2_hat)
    cooks_d = cook_distance(std_residuals, h_ii, p)
    
    y_hat = manual_matmul(X_list, beta_hat)
    
    sqrt_abs_std_resid = [math.sqrt(abs(r)) for r in std_residuals]
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Biểu đồ 1: Residuals vs Fitted (Kiểm tra tính tuyến tính và đồng phương sai)
    axs[0, 0].scatter(y_hat, residuals, alpha=0.5, edgecolors='k')
    axs[0, 0].axhline(0, color='red', linestyle='--')
    axs[0, 0].set_title('Residuals vs Fitted')
    axs[0, 0].set_xlabel('Fitted values')
    axs[0, 0].set_ylabel('Residuals')
    axs[0, 0].grid(True, alpha=0.3)
    
    # Biểu đồ 2: Normal Q-Q (Kiểm tra phân phối chuẩn của sai số)
    stats.probplot(std_residuals, dist="norm", plot=axs[0, 1])
    axs[0, 1].set_title('Normal Q-Q')
    axs[0, 1].grid(True, alpha=0.3)
    
    # Biểu đồ 3: Scale-Location (Kiểm tra tính đồng nhất của phương sai)
    axs[1, 0].scatter(y_hat, sqrt_abs_std_resid, alpha=0.5, edgecolors='k')
    axs[1, 0].set_title('Scale-Location')
    axs[1, 0].set_xlabel('Fitted values')
    axs[1, 0].set_ylabel('$\\sqrt{|Standardized Residuals|}$')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Biểu đồ 4: Cook's Distance (Phát hiện các điểm ảnh hưởng lớn)
    axs[1, 1].stem(list(range(len(cooks_d))), cooks_d, markerfmt=",")
    axs[1, 1].axhline(4/n, color='red', linestyle='--', label=f'Threshold 4/n ({round(4/n,3)})')
    axs[1, 1].set_title("Cook's Distance")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()