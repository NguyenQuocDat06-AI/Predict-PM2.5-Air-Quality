
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
    """Giải hệ phương trình Ax = b bằng phương pháp Khử Gauss"""
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

def ridge_fit(X, y, alpha):
    """
    Tính ước lượng Ridge: beta_ridge = (X^T X + alpha * I)^-1 X^T y
    """
    X_list = X.tolist() if hasattr(X, "tolist") else X
    y_list = y.tolist() if hasattr(y, "tolist") else y
    p = len(X_list[0])
    
    Xt = manual_transpose(X_list)
    XtX = manual_matmul(Xt, X_list)
    
    # Cộng alpha * I vào đường chéo
    for i in range(p):
        XtX[i][i] += alpha
        
    Xty = manual_matmul(Xt, y_list)
    beta_ridge = manual_solve(XtX, Xty)
    
    return beta_ridge

def ridge_trace(X, y, alphas):
    """Tính danh sách các hệ số Ridge ứng với tập alpha"""
    traces = []
    for a in alphas:
        traces.append(ridge_fit(X, y, a))
    return traces