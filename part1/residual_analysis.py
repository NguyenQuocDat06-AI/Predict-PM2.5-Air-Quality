import math

def manual_transpose(M):
    """Tính ma trận chuyển vị M^T"""
    rows = len(M)
    cols = len(M[0])
    return [[M[j][i] for j in range(rows)] for i in range(cols)]

def manual_matmul(A, B):
    """Nhân hai ma trận A * B (hoặc ma trận * vector)"""
    rows_A = len(A)
    cols_A = len(A[0])
    if not isinstance(B[0], list):
        res = [0.0] * rows_A
        for i in range(rows_A):
            for k in range(cols_A):
                res[i] += A[i][k] * B[k]
        return res
    else:
        rows_B = len(B)
        cols_B = len(B[0])
        res = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    res[i][j] += A[i][k] * B[k][j]
        return res

def manual_solve(A, b):
    n = len(A)
    M = [row[:] + [val] for row, val in zip(A, b)]
    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > abs(M[max_row][i]):
                max_row = k
        M[i], M[max_row] = M[max_row], M[i]
        for j in range(i + 1, n):
            ratio = M[j][i] / M[i][i]
            for k in range(i, n + 1):
                M[j][k] -= ratio * M[i][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (M[i][n] - s) / M[i][i]
    return x

def manual_inv(M):
    n = len(M)
    res = [[0.0 for _ in range(n)] for _ in range(n)]
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        col = manual_solve(M, e)
        for i in range(n):
            res[i][j] = col[i]
    return res

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