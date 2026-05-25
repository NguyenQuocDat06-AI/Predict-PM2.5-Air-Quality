import numpy as np
import matplotlib.pyplot as plt

def ridge_fit(X, y, lam_list):
    """
    Cài đặt Ridge Regression: beta_ridge = (X^T X + lambda*I)^-1 X^T y
    Vẽ Ridge Trace biểu diễn sự thay đổi của các hệ số theo lambda.
    """
    n, p = X.shape
    has_intercept = np.allclose(X[:, 0], 1)
    
    beta_hats = []
    
    for l in lam_list:
        # Ma trận đơn vị cho phần phạt (penalty)
        I = np.eye(p)
        if has_intercept:
            I[0, 0] = 0 # Không phạt hệ số chặn (intercept)
            
        # Giải hệ phương trình (X^T X + lambda*I) beta = X^T y
        A = X.T @ X + l * I
        b = X.T @ y
        beta = np.linalg.solve(A, b)
        beta_hats.append(beta)
        
    beta_hats = np.array(beta_hats)
    
    # Vẽ Ridge Trace
    plt.figure(figsize=(10, 6))
    start_idx = 1 if has_intercept else 0
    for j in range(start_idx, p):
        plt.plot(lam_list, beta_hats[:, j], label=f'beta_{j}')
        
    plt.xscale('log')
    plt.xlabel('Giá trị Lambda (log scale)')
    plt.ylabel('Giá trị hệ số hồi quy (beta)')
    plt.title('Biểu đồ Ridge Trace')
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return beta_hats

if __name__ == "__main__":
    # Demo Ridge và Lasso với dữ liệu giả lập
    np.random.seed(42)
    n, p = 100, 10
    X = np.random.randn(n, p)
    # Thêm intercept
    X_with_intercept = np.hstack([np.ones((n, 1)), X])
    true_beta = np.random.randn(p + 1)
    y = X_with_intercept @ true_beta + np.random.normal(0, 1, n)
    
    lam_list = np.logspace(-2, 4, 100)
    
    print("Đang vẽ Ridge Trace...")
    ridge_fit(X_with_intercept, y, lam_list)
    
    print("Đang vẽ Lasso Trace...")
    lasso_trace(X_with_intercept, y, lam_list)

def lasso_fit(X, y, lam, n_iters=1000, tol=1e-4):
    """
    Cài đặt Lasso Regression sử dụng thuật toán Coordinate Descent.
    Nghiệm của Lasso không có dạng closed-form.
    """
    n, p = X.shape
    beta = np.zeros(p)
    
    # Khởi tạo beta bằng OLS hoặc 0
    # Ở đây dùng 0 để minh họa thuật toán co rút (shrinkage)
    
    for _ in range(n_iters):
        beta_old = beta.copy()
        
        for j in range(p):
            # Tính phần dư mà không dùng biến j hiện tại
            y_pred_no_j = X @ beta - X[:, j] * beta[j]
            rho_j = X[:, j] @ (y - y_pred_no_j)
            
            # Tính chuẩn (norm) của cột X_j
            norm_j = np.sum(X[:, j]**2)
            
            if j == 0 and np.allclose(X[:, 0], 1): # Intercept
                beta[j] = rho_j / n
            else:
                # Soft Thresholding
                beta[j] = np.sign(rho_j) * max(abs(rho_j) - lam, 0) / (norm_j if norm_j != 0 else 1)
                
        # Kiểm tra hội tụ
        if np.linalg.norm(beta - beta_old) < tol:
            break
            
    return beta

def lasso_trace(X, y, lam_list):
    """
    Vẽ Lasso Trace biểu diễn sự thay đổi của các hệ số theo lambda.
    """
    n, p = X.shape
    beta_hats = []
    
    for l in lam_list:
        beta = lasso_fit(X, y, l)
        beta_hats.append(beta)
        
    beta_hats = np.array(beta_hats)
    
    plt.figure(figsize=(10, 6))
    has_intercept = np.allclose(X[:, 0], 1)
    start_idx = 1 if has_intercept else 0
    
    for j in range(start_idx, p):
        plt.plot(lam_list, beta_hats[:, j], label=f'beta_{j}')
        
    plt.xscale('log')
    plt.xlabel('Giá trị Lambda (log scale)')
    plt.ylabel('Giá trị hệ số hồi quy (beta)')
    plt.title('Biểu đồ Lasso Trace')
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return beta_hats