from ols_implementation import manual_transpose, manual_matmul, manual_solve

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