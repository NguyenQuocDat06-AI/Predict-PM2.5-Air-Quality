import random
from ols_implementation import ols_fit, manual_matmul

def kfold_cv(X, y, k, random_state=42):
    """
    k-Fold CV sử dụng OLS.
    """
    X_list = X.tolist() if hasattr(X, "tolist") else X
    y_list = y.tolist() if hasattr(y, "tolist") else y
    n = len(y_list)
    
    # Tạo danh sách chỉ số và trộn ngẫu nhiên
    indices = list(range(n))
    random.seed(random_state)
    random.shuffle(indices)

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

        # Thực hiện OLS trên fold (sử dụng ols_fit đã có pivoting)
        beta_hat, _ = ols_fit(X_train, y_train)

        # Tính MSE
        y_pred = manual_matmul(X_test, beta_hat)
        mse = sum((y_test[m] - y_pred[m])**2 for m in range(len(y_test))) / len(y_test)
        mse_list.append(mse)

    avg_mse = sum(mse_list) / len(mse_list)
    return avg_mse, mse_list