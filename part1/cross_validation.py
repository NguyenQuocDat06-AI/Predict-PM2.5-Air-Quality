
def manual_transpose(M):
    """Tính ma trận chuyển vị M^T bằng vòng lặp"""
    rows = len(M)
    cols = len(M[0])
    return [[M[j][i] for j in range(rows)] for i in range(cols)]

def manual_matmul(A, B):
    """Nhân hai ma trận A * B (hoặc ma trận * vector) bằng vòng lặp"""
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

def kfold_cv(X, y, k, random_state=42):
    """
    k-Fold CV sử dụng OLS thuần list.
    """
    import numpy as np # Dùng để chia fold (sinh indices ngẫu nhiên)
    X_list = X.tolist() if hasattr(X, "tolist") else X
    y_list = y.tolist() if hasattr(y, "tolist") else y
    n = len(y_list)
    
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n).tolist()

    # Chia fold
    folds = []
    fold_size = n // k
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i != k - 1 else n
        folds.append(indices[start:end])

    mse_list = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = []
        for j in range(k):
            if j != i:
                train_idx.extend(folds[j])

        # Tạo tập Train/Test theo indices
        X_train = [X_list[idx] for idx in train_idx]
        y_train = [y_list[idx] for idx in train_idx]
        
        X_test = [X_list[idx] for idx in test_idx]
        y_test = [y_list[idx] for idx in test_idx]

        # Thực hiện OLS trên fold
        Xt = manual_transpose(X_train)
        XtX = manual_matmul(Xt, X_train)
        Xty = manual_matmul(Xt, y_train)
        beta_hat = manual_solve(XtX, Xty)

        # Tính MSE
        y_pred = manual_matmul(X_test, beta_hat)
        mse = sum((y_test[m] - y_pred[m])**2 for m in range(len(y_test))) / len(y_test)
        mse_list.append(mse)

    avg_mse = sum(mse_list) / len(mse_list)
    return avg_mse, mse_list