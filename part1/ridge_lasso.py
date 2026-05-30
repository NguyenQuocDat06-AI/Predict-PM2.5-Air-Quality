from ols_implementation import manual_transpose, manual_matmul, manual_solve

def ridge_fit(X, y, alpha):
    """
    Tính ước lượng Ridge Regression:
    beta_ridge = (X^T X + alpha * I)^{-1} X^T y

    Khác với OLS, Ridge thêm thành phần điều chỉnh alpha * I vào X^T X
    để xử lý đa cộng tuyến và tránh overfitting.

    Parameters
    ----------
    X     : list[list[float]]  — ma trận thiết kế kích thước (n x (p+1)),
                                 cột đầu tiên là intercept (toàn giá trị 1)
    y     : list[float]        — vector phản hồi quan sát kích thước (n,)
    alpha : float              — hệ số điều chỉnh (regularization strength),
                                 alpha = 0 tương đương OLS thông thường,
                                 alpha càng lớn thì hệ số càng bị co về 0

    Returns
    -------
    beta_ridge : list[float]  — vector hệ số Ridge ước lượng kích thước (p+1,)
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
    """
    Tính danh sách các vector hệ số Ridge ứng với từng giá trị alpha,
    dùng để vẽ Ridge Trace (biểu đồ theo dõi sự thay đổi hệ số theo alpha).

    Parameters
    ----------
    X      : list[list[float]]  — ma trận thiết kế kích thước (n x (p+1))
    y      : list[float]        — vector phản hồi quan sát kích thước (n,)
    alphas : list[float]        — danh sách các giá trị alpha cần thử nghiệm,
                                  thường được lấy theo thang log
                                  (ví dụ: [0.001, 0.01, 0.1, 1, 10, 100])

    Returns
    -------
    traces : list[list[float]]  — danh sách các vector beta_ridge tương ứng,
                                  kích thước (len(alphas) x (p+1))
    """
    traces = []
    for a in alphas:
        traces.append(ridge_fit(X, y, a))
    return traces