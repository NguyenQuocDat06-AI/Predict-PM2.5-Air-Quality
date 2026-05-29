import math
from ols_implementation import manual_transpose, manual_matmul, manual_solve, manual_inv

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