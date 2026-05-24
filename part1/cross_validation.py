import numpy as np

def kfold_cv(X, y, k, random_state=42):
    """
    k-Fold Cross-Validation, tính CV Score = mean MSE (Sai số bình phương trung bình).
    """
    n = len(y)
    rng = np.random.default_rng(random_state)  # Đảm bảo kết quả có thể tái lập
    indices = rng.permutation(n)

    # Chia thành k fold bằng nhau
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[:n % k] += 1

    folds = []
    current = 0
    for size in fold_sizes:
        folds.append(indices[current:current + size])
        current += size

    mse_list = []
    for i in range(k):
        test_idx  = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        # OLS từ đầu
        XtX = X_train.T @ X_train
        Xty = X_train.T @ y_train
        beta_hat = np.linalg.solve(XtX, Xty)

        y_pred = X_test @ beta_hat
        mse_list.append(np.mean((y_test - y_pred) ** 2))

    cv_score = np.mean(mse_list)

    print(f"=== Kết quả {k}-Fold CV ===")
    for i, mse in enumerate(mse_list):
        print(f"  Fold {i+1}: MSE = {mse:.4f}")
    print(f"  CV Score (Mean MSE) = {cv_score:.4f}")

    return cv_score, mse_list

if __name__ == "__main__":
    # Demo k-Fold Cross-Validation với dữ liệu giả lập
    np.random.seed(42)
    n, p = 150, 5
    X_base = np.random.randn(n, p)
    X = np.hstack([np.ones((n, 1)), X_base])
    true_beta = np.array([5, 2, -3, 1, 0, -1])
    y = X @ true_beta + np.random.normal(0, 2, n)
    
    # Chạy CV với k=5
    kfold_cv(X, y, k=5)
    
    # Thử với k=10
    kfold_cv(X, y, k=10)