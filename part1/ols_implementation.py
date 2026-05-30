import math
from scipy import stats

def manual_transpose(M):
    """Tính ma trận chuyển vị M^T bằng vòng lặp"""
    rows = len(M)
    cols = len(M[0])
    return [[M[j][i] for j in range(rows)] for i in range(cols)]

def manual_matmul(A, B):
    """Nhân hai ma trận A * B (hoặc ma trận * vector) bằng vòng lặp"""
    rows_A = len(A)
    cols_A = len(A[0])
    
    # Kiểm tra B là vector hay ma trận
    if not isinstance(B[0], list):
        # B là vector 1 chiều
        if cols_A != len(B):
            raise ValueError("Kích thước không khớp")
        res = [0.0] * rows_A
        for i in range(rows_A):
            for k in range(cols_A):
                res[i] += A[i][k] * B[k]
        return res
    else:
        # B là ma trận 2 chiều
        rows_B = len(B)
        cols_B = len(B[0])
        if cols_A != rows_B:
            raise ValueError("Kích thước không khớp")
        res = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    res[i][j] += A[i][k] * B[k][j]
        return res

def manual_solve(A, b):
    """
    Giải hệ phương trình Ax = b bằng phương pháp Khử Gauss (Gaussian Elimination)
    Dùng Partial Pivoting để tăng độ chính xác.
    """
    n = len(A)
    # Tạo ma trận mở rộng [A | b]
    M = [row[:] + [val] for row, val in zip(A, b)]

    # 1. Khử xuôi
    for i in range(n):
        # Chọn trục xoay
        max_row = i
        max_val = abs(M[i][i])
        for k in range(i + 1, n):
            if abs(M[k][i]) > max_val:
                max_val = abs(M[k][i])
                max_row = k
        M[i], M[max_row] = M[max_row], M[i]

        if abs(M[i][i]) < 1e-15:
            # Nếu hệ vô nghiệm hoặc vô số nghiệm, trả về vector 0 hoặc báo lỗi
            raise ValueError("Ma trận suy biến")

        for j in range(i+1, n):
            ratio = M[j][i] / M[i][i]
            for k in range(i, n + 1):
                M[j][k] -= ratio * M[i][k]

    # 2. Thế ngược
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (M[i][n] - s) / M[i][i]
    return x

def manual_inv(M):
    """Nghịch đảo ma trận bằng cách giải n hệ phương trình Ax = e_i"""
    n = len(M)
    res = [[0.0 for _ in range(n)] for _ in range(n)]
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        col_sol = manual_solve(M, e)
        for i in range(n):
            res[i][j] = col_sol[i]
    return res

def ols_fit(X, y):
    """Tính beta_hat = (X^T X)^-1 X^T y"""
    # Chuyển ndarray về list
    X_list = X.tolist() if hasattr(X, "tolist") else X
    y_list = y.tolist() if hasattr(y, "tolist") else y
    
    n = len(X_list)
    p_plus_1 = len(X_list[0])
    
    Xt = manual_transpose(X_list)
    XtX = manual_matmul(Xt, X_list)
    Xty = manual_matmul(Xt, y_list)
    
    beta_hat = manual_solve(XtX, Xty)
    
    # Tính y_hat và RSS
    y_hat = manual_matmul(X_list, beta_hat)
    rss = sum((y_list[i] - y_hat[i])**2 for i in range(n))
    sigma2_hat = rss / (n - p_plus_1)
    
    return beta_hat, sigma2_hat

def hat_matrix(X):
    """Tính ma trận chiếu H = X(X^T X)^-1 X^T"""
    X_list = X.tolist() if hasattr(X, "tolist") else X
    Xt = manual_transpose(X_list)
    XtX = manual_matmul(Xt, X_list)
    XtX_inv = manual_inv(XtX)
    
    H = manual_matmul(manual_matmul(X_list, XtX_inv), Xt)
    return H

def model_metrics(y, y_hat, p):
    """Tính RSS, TSS, R^2, R^2_adj, F_stat"""
    y_list = y.tolist() if hasattr(y, "tolist") else y
    y_hat_list = y_hat.tolist() if hasattr(y_hat, "tolist") else y_hat
    n = len(y_list)
    
    rss = sum((y_list[i] - y_hat_list[i])**2 for i in range(n))
    y_mean = sum(y_list) / n
    tss = sum((yi - y_mean)**2 for yi in y_list)
    
    r2 = 1 - (rss / tss) if tss != 0 else 0
    r2_adj = 1 - ((n - 1) / (n - p - 1)) * (1 - r2)
    
    if rss > 1e-15:
        f_stat = ((tss - rss) / p) / (rss / (n - p - 1))
    else:
        f_stat = float('inf')
    f_pvalue = 1 - stats.f.cdf(f_stat, dfn=p, dfd=n - p - 1) if f_stat != float('inf') else 0.0
    
    return rss, tss, r2, r2_adj, f_stat, f_pvalue

def coef_inference(X, y, beta_hat, sigma2_hat):
    """Tính SE, t-stat, p-value và CI cho các hệ số"""
    X_list = X.tolist() if hasattr(X, "tolist") else X
    n = len(X_list)
    p_plus_1 = len(X_list[0])
    
    Xt = manual_transpose(X_list)
    XtX = manual_matmul(Xt, X_list)
    XtX_inv = manual_inv(XtX)
    
    se = []
    for i in range(p_plus_1):
        var_bi = sigma2_hat * XtX_inv[i][i]
        se.append(math.sqrt(max(var_bi, 0)))
        
    t_stat = []
    for i in range(p_plus_1):
        if se[i] > 1e-15:
            t_stat.append(beta_hat[i] / se[i])
        else:
            t_stat.append(float('inf') if beta_hat[i] != 0 else 0)
            
    df = n - p_plus_1
    p_values = [2 * (1 - stats.t.cdf(abs(t), df)) for t in t_stat]
    
    t_crit = stats.t.ppf(0.975, df)
    ci_lower = [beta_hat[i] - t_crit * se[i] for i in range(p_plus_1)]
    ci_upper = [beta_hat[i] + t_crit * se[i] for i in range(p_plus_1)]
    
    return se, t_stat, p_values, (ci_lower, ci_upper)

def vif(X):
    """Tính chuẩn VIF cho từng biến"""
    X_list = X.tolist() if hasattr(X, "tolist") else X
    n = len(X_list)
    p = len(X_list[0])
    
    # Kiểm tra intercept
    has_intercept = all(abs(row[0] - 1.0) < 1e-9 for row in X_list)
    start_idx = 1 if has_intercept else 0
    
    vif_vals = []
    for j in range(start_idx, p):
        # Biến j là target, các biến còn lại là predictors
        X_other = []
        y_j = []
        for row in X_list:
            X_other.append([row[k] for k in range(p) if k != j])
            y_j.append(row[j])
            
        # OLS
        Xt_other = manual_transpose(X_other)
        A = manual_matmul(Xt_other, X_other)
        B = manual_matmul(Xt_other, y_j)
        b_hat = manual_solve(A, B)
        
        y_j_hat = manual_matmul(X_other, b_hat)
        rss = sum((y_j[i] - y_j_hat[i])**2 for i in range(len(y_j)))
        y_mean = sum(y_j) / len(y_j)
        tss = sum((yj - y_mean)**2 for yj in y_j)
        
        r2_j = 1 - (rss / tss) if tss > 1e-12 else 0
        vif_vals.append(1 / (1 - min(r2_j, 0.999999)))
        
    return vif_vals

def run_monte_carlo(X, true_beta, sigma=1.0, n_sims=1000, seed=42):
    """Mô phỏng Monte Carlo kiểm chứng BLUE (Gauss-Markov)"""
    import random
    random.seed(seed)
    X_list = X.tolist() if hasattr(X, "tolist") else X
    p = len(X_list[0])
    
    Xt = manual_transpose(X_list)
    XtX_inv = manual_inv(manual_matmul(Xt, X_list))
    W_ols = manual_matmul(XtX_inv, Xt)
    
    ols_betas = []
    for _ in range(n_sims):
        eps = [random.gauss(0, sigma) for _ in range(len(X_list))]
        # y = X*beta + eps
        y = [manual_matmul([X_list[i]], true_beta)[0] + eps[i] for i in range(len(X_list))]
        b_ols = manual_matmul(W_ols, y)
        ols_betas.append(b_ols)
        
    return ols_betas


