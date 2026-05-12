import numpy as np
from scipy import stats


def ols_fit(X, y):
    # TODO: Cài đặt công thức tính beta_hat và sigma2_hat
    beta_hat = np.zeros(X.shape[1])
    sigma2_hat = 0.0
    return beta_hat, sigma2_hat


def hat_matrix(X):
    # TODO: Cài đặt công thức ma trận chiếu H
    H = np.eye(X.shape[0])
    return H


def model_metrics(y, y_hat, p):
    # TODO: Tính RSS, TSS, R^2, R^2_adj, F_stat
    return 0.0, 0.0, 0.0, 0.0, 0.0


def coef_inference(X, y, beta_hat, sigma2_hat):
    # TODO: Tính se, t_stat, p_value, khoảng tin cậy
    n_features = len(beta_hat)
    return (
        np.zeros(n_features),
        np.zeros(n_features),
        np.zeros(n_features),
        (np.zeros(n_features), np.zeros(n_features)),
    )
