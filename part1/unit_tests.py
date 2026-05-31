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
        # Case 1: 2x2 matrix
        m1 = [[1, 2], [3, 4]]
        self.assertEqual(manual_transpose(m1), [[1, 3], [2, 4]])
        # Case 2: 2x3 matrix
        m2 = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(manual_transpose(m2), [[1, 4], [2, 5], [3, 6]])

    def test_manual_matmul(self):
        # Case 1: Matrix and vector
        A = [[1, 2], [3, 4]]
        v = [5, 6]
        self.assertEqual(manual_matmul(A, v), [17, 39])
        # Case 2: Matrix and Identity matrix
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
        # Case 1: Identity matrix inverse
        I = [[1, 0], [0, 1]]
        self.assertEqual(manual_inv(I), I)
        # Case 2: Simple 2x2 matrix inverse
        A = [[4, 7], [2, 6]]
        invA = manual_inv(A)
        self.assertAlmostEqual(invA[0][0], 0.6)
        self.assertAlmostEqual(invA[0][1], -0.7)
        self.assertAlmostEqual(invA[1][0], -0.2)
        self.assertAlmostEqual(invA[1][1], 0.4)

    def test_ols_fit(self):
        # Case 1: Perfect line y = 1 + 2x
        X = [[1, 1], [1, 2], [1, 3], [1, 4]]
        y = [3, 5, 7, 9]
        beta, sigma2 = ols_fit(X, y)
        self.assertAlmostEqual(beta[0], 1.0)
        self.assertAlmostEqual(beta[1], 2.0)
        self.assertAlmostEqual(sigma2, 0.0)

        # Case 2: Data with noise
        X_n = [[1, 0], [1, 1], [1, 2], [1, 3]]
        y_n = [1.1, 1.9, 3.1, 3.9] 
        beta_n, sigma2_n = ols_fit(X_n, y_n)
        # Expected OLS slope beta_1 = 0.96, beta_0 = 1.06
        self.assertAlmostEqual(beta_n[0], 1.06)
        self.assertAlmostEqual(beta_n[1], 0.96)
        self.assertAlmostEqual(sigma2_n, 0.016, places=3)

    def test_hat_matrix(self):
        # Case 1: Small design matrix
        X = [[1, 1], [1, 2], [1, 3]]
        H = hat_matrix(X)
        trace = sum(H[i][i] for i in range(len(H)))
        self.assertAlmostEqual(trace, 2.0) # trace = p+1 = 2
        H2 = manual_matmul(H, H)
        for i in range(len(H)):
            for j in range(len(H[0])):
                self.assertAlmostEqual(H[i][j], H2[i][j]) # Idempotent property: H^2 = H

        # Case 2: Another design matrix
        X2 = [[1, 0], [1, 5]]
        H3 = hat_matrix(X2)
        # For full rank square matrix, hat matrix should be Identity matrix
        self.assertAlmostEqual(H3[0][0], 1.0)
        self.assertAlmostEqual(H3[0][1], 0.0)
        self.assertAlmostEqual(H3[1][1], 1.0)

    def test_coef_inference(self):
        # Case 1: Simple regression with noise
        X = [[1, 0], [1, 1], [1, 2], [1, 3]]
        y = [1.1, 1.9, 3.1, 3.9]
        beta_hat, sigma2_hat = ols_fit(X, y)
        se, t_stat, p_values, ci = coef_inference(X, y, beta_hat, sigma2_hat)
        # Check that standard errors are strictly positive
        self.assertGreater(se[0], 0.0)
        self.assertGreater(se[1], 0.0)
        # CI Lower bound should be less than Upper bound
        self.assertLess(ci[0][0], ci[1][0])

        # Case 2: Perfect fit (no noise), standard errors should be 0
        X2 = [[1, 1], [1, 2], [1, 3]]
        y2 = [2.0, 4.0, 6.0]
        beta_hat2, sigma2_hat2 = ols_fit(X2, y2)
        se2, t_stat2, p_values2, ci2 = coef_inference(X2, y2, beta_hat2, sigma2_hat2)
        self.assertAlmostEqual(se2[0], 0.0)
        self.assertAlmostEqual(se2[1], 0.0)

    def test_model_metrics(self):
        # Case 1: Perfect fit y = y_hat
        y = [3, 5, 7]
        y_hat = [3, 5, 7]
        rss, tss, r2, r2_adj, f_stat, f_pval = model_metrics(y, y_hat, 1)
        self.assertAlmostEqual(rss, 0.0)
        self.assertAlmostEqual(r2, 1.0)
        self.assertEqual(f_stat, float('inf'))

        # Case 2: Realistic noisy fit
        y2 = [1.0, 2.0, 3.0]
        y_hat2 = [1.1, 1.9, 3.0]
        rss2, tss2, r22, r2_adj2, f_stat2, f_pval2 = model_metrics(y2, y_hat2, 1)
        # TSS = sum((yi - 2)^2) = 1 + 0 + 1 = 2.0
        # RSS = sum((yi - y_hat_i)^2) = 0.01 + 0.01 + 0.0 = 0.02
        # R2 = 1 - 0.02/2.0 = 0.99
        self.assertAlmostEqual(tss2, 2.0)
        self.assertAlmostEqual(rss2, 0.02)
        self.assertAlmostEqual(r22, 0.99)
        self.assertGreater(f_stat2, 0.0)

    def test_vif(self):
        # Case 1: Completely orthogonal features
        X = [[1, 1, 0], [1, 0, 1], [1, -1, 0], [1, 0, -1]]
        vif_vals = vif(X)
        for val in vif_vals:
            self.assertAlmostEqual(val, 1.0, places=3)

        # Case 2: Highly collinear features
        X2 = [[1, 1, 1.01], [1, 2, 2.02], [1, 3, 3.03], [1, 4, 4.04]]
        vif_vals2 = vif(X2)
        # Variable 1 and 2 are almost identical, VIF should be extremely high
        self.assertGreater(vif_vals2[0], 100.0)
        self.assertGreater(vif_vals2[1], 100.0)

    # --- Tests for ridge_lasso.py ---

    def test_ridge_fit(self):
        # Case 1: Simple OLS vs Ridge comparison (alpha = 10)
        X = [[1, 1], [1, 2]]
        y = [2, 4]
        beta_ols = [0.0, 2.0]
        beta_ridge = ridge_fit(X, y, alpha=10)
        # Ridge coefficients should be shrunk toward 0
        self.assertLess(abs(beta_ridge[1]), abs(beta_ols[1]))

        # Case 2: Ridge with alpha = 0 should equal OLS fit
        beta_ridge_0 = ridge_fit(X, y, alpha=0)
        self.assertAlmostEqual(beta_ridge_0[0], beta_ols[0])
        self.assertAlmostEqual(beta_ridge_0[1], beta_ols[1])

    # --- Tests for residual_analysis.py ---

    def test_get_residuals(self):
        # Case 1: Simple subtraction check
        X = [[1, 1], [1, 2]]
        y = [2, 5]
        beta = [0, 2]
        res = get_residuals(X, y, beta)
        self.assertEqual(res, [2 - 2, 5 - 4]) # [0, 1]

        # Case 2: Check another set
        res2 = get_residuals([[1, 3]], [10], [1, 2])
        self.assertEqual(res2, [10 - 7]) # [3]

    def test_cook_distance(self):
        # Case 1: Standard calculation check
        std_res = [2.0]
        h = [0.5]
        p = 2 # total parameters
        d = cook_distance(std_res, h, p)
        # Expected: (4 / 2) * (0.5 / 0.5) = 2 * 1 = 2.0
        self.assertAlmostEqual(d[0], 2.0)

        # Case 2: Larger inputs
        std_res2 = [1.0, 3.0]
        h2 = [0.2, 0.1]
        d2 = cook_distance(std_res2, h2, p)
        # Expected 0: (1.0 / 2) * (0.2 / 0.8) = 0.5 * 0.25 = 0.125
        # Expected 1: (9.0 / 2) * (0.1 / 0.9) = 4.5 * (1/9) = 0.5
        self.assertAlmostEqual(d2[0], 0.125)
        self.assertAlmostEqual(d2[1], 0.5)

    # --- Tests for cross_validation.py ---

    def test_kfold_cv(self):
        # Case 1: Perfect fit on clean linear relationship
        X = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]]
        y = [2, 4, 6, 8, 10, 12]
        avg_mse, mse_list = kfold_cv(X, y, k=2)
        self.assertAlmostEqual(avg_mse, 0.0, places=5)

        # Case 2: Simple noisy set with k=3
        X2 = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]]
        y2 = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        avg_mse2, mse_list2 = kfold_cv(X2, y2, k=3)
        self.assertEqual(len(mse_list2), 3)
        self.assertGreaterEqual(avg_mse2, 0.0)

if __name__ == '__main__':
    unittest.main()
