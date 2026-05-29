import unittest
import math
from ols_implementation import (
    manual_transpose, manual_matmul, manual_solve, manual_inv, 
    ols_fit, model_metrics, vif, hat_matrix, coef_inference
)
from ridge_lasso import ridge_fit
from residual_analysis import get_residuals, cook_distance
from cross_validation import kfold_cv

class TestPart1(unittest.TestCase):
    
    # --- Tests for ols_implementation.py ---
    
    def test_manual_transpose(self):
        m1 = [[1, 2], [3, 4]]
        self.assertEqual(manual_transpose(m1), [[1, 3], [2, 4]])
        m2 = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(manual_transpose(m2), [[1, 4], [2, 5], [3, 6]])

    def test_manual_matmul(self):
        A = [[1, 2], [3, 4]]
        v = [5, 6]
        self.assertEqual(manual_matmul(A, v), [17, 39])
        B = [[1, 0], [0, 1]]
        self.assertEqual(manual_matmul(A, B), A)

    def test_manual_solve(self):
        # Case 1: Simple 2x2
        A = [[2, 1], [1, 1]]
        b = [5, 3] # solution x=[2, 1]
        x = manual_solve(A, b)
        self.assertAlmostEqual(x[0], 2.0)
        self.assertAlmostEqual(x[1], 1.0)
        # Case 2: 3x3 diagonal
        A2 = [[2, 0, 0], [0, 3, 0], [0, 0, 4]]
        b2 = [2, 6, 12]
        x2 = manual_solve(A2, b2)
        self.assertEqual(x2, [1.0, 2.0, 3.0])

    def test_manual_inv(self):
        I = [[1, 0], [0, 1]]
        self.assertEqual(manual_inv(I), I)
        A = [[4, 7], [2, 6]]
        invA = manual_inv(A)
        self.assertAlmostEqual(invA[0][0], 0.6)
        self.assertAlmostEqual(invA[0][1], -0.7)

    def test_ols_fit(self):
        # Case 1: Perfect line
        X = [[1, 1], [1, 2], [1, 3], [1, 4]]
        y = [3, 5, 7, 9]
        beta, sigma2 = ols_fit(X, y)
        self.assertAlmostEqual(beta[1], 2.0)
        self.assertAlmostEqual(sigma2, 0.0)

        # Case 2: Data with noise
        X_n = [[1, 0], [1, 1], [1, 2], [1, 3]]
        y_n = [1.1, 1.9, 3.1, 3.9] 
        beta_n, sigma2_n = ols_fit(X_n, y_n)
        # RSS = 0.032, df = 2 => Sigma2 = 0.016
        self.assertAlmostEqual(sigma2_n, 0.016, places=3)

    def test_hat_matrix(self):
        X = [[1, 1], [1, 2], [1, 3]]
        H = hat_matrix(X)
        trace = sum(H[i][i] for i in range(len(H)))
        self.assertAlmostEqual(trace, 2.0) # p+1
        H2 = manual_matmul(H, H)
        for i in range(len(H)):
            for j in range(len(H[0])):
                self.assertAlmostEqual(H[i][j], H2[i][j])

    def test_coef_inference(self):
        # Use noise so SE != 0 to test stability
        X = [[1, 0], [1, 1], [1, 2], [1, 3]]
        y = [1.1, 1.9, 3.1, 3.9]
        beta_hat, sigma2_hat = ols_fit(X, y)
        se, t_stat, p_values, ci = coef_inference(X, y, beta_hat, sigma2_hat)
        # CI Lower bound should be less than Upper bound
        self.assertLess(ci[0][0], ci[1][0])

    def test_model_metrics(self):
        y = [3, 5, 7]
        y_hat = [3, 5, 7]
        rss, tss, r2, r2_adj, f_stat, f_pval = model_metrics(y, y_hat, 1)
        self.assertAlmostEqual(rss, 0.0)
        self.assertEqual(f_stat, float('inf'))

    def test_vif(self):
        # Case 1: Orthogonal features
        X = [[1, 1, 0], [1, 0, 1], [1, -1, 0], [1, 0, -1]]
        vif_vals = vif(X)
        for val in vif_vals:
            self.assertAlmostEqual(val, 1.0, places=3) # Reduced places for stability

    # --- Tests for ridge_lasso.py ---

    def test_ridge_fit(self):
        X = [[1, 1], [1, 2]]
        y = [2, 4]
        beta_ols = [0, 2]
        beta_ridge = ridge_fit(X, y, alpha=10)
        self.assertLess(abs(beta_ridge[1]), abs(beta_ols[1]))

    # --- Tests for residual_analysis.py ---

    def test_get_residuals(self):
        X = [[1, 1], [1, 2]]
        y = [2, 5]
        beta = [0, 2]
        res = get_residuals(X, y, beta)
        self.assertEqual(res, [0, 1])

    def test_cook_distance(self):
        # Formula: Di = (std_res^2 / p) * (h / (1-h))
        std_res = [2.0]
        h = [0.5]
        p = 2 # total parameters
        d = cook_distance(std_res, h, p)
        # Expected: (4 / 2) * (0.5 / 0.5) = 2 * 1 = 2.0
        self.assertAlmostEqual(d[0], 2.0)

    # --- Tests for cross_validation.py ---

    def test_kfold_cv(self):
        X = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]]
        y = [2, 4, 6, 8, 10, 12]
        avg_mse, mse_list = kfold_cv(X, y, k=2)
        # Check precision explicitly
        self.assertAlmostEqual(avg_mse, 0.0, places=5)

if __name__ == '__main__':
    unittest.main()
