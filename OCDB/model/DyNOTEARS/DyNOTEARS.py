from ..BaseModel import BaseModel
import numpy as np
import scipy.optimize as sopt
from .utils import _reshape_wa
import warnings
import scipy.linalg as slin
import networkx as nx


class DyNOTEARS(BaseModel):

    def __init__(self, device, p, lambda_w: float = 0.1, lambda_a: float = 0.1,
                 max_iter: int = 10, h_tol: float = 1e-8, w_threshold: float = 0.0):
        super().__init__(device)
        self.p = p
        self.lambda_w = lambda_w
        self.lambda_a = lambda_a
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.w_threshold = w_threshold

    def fit(self, feature):
        X = feature[self.p:]
        X_lags = feature[:feature.shape[0] - self.p]


        n, d_vars = X.shape
        p_orders = X_lags.shape[1] // d_vars
        wa_est = np.zeros(2 * (p_orders + 1) * d_vars**2)
        wa_new = np.zeros(2 * (p_orders + 1) * d_vars**2)
        rho, alpha, h_value, h_new = 1.0, 0.0, np.inf, np.inf

        def _h(wa_vec: np.ndarray) -> float:
            """
            Constraint function of the dynotears

            Args:
                wa_vec (np.ndarray): current adjacency vector with intra- and inter-slice weights

            Returns:
                float: DAGness of the intra-slice adjacency matrix W (0 == DAG, >0 == cyclic)
            """

            _w_mat, _ = _reshape_wa(wa_vec, d_vars, p_orders)
            return np.trace(slin.expm(_w_mat * _w_mat)) - d_vars
        
        def _func(wa_vec: np.ndarray) -> float:
            """
            Objective function that the dynotears tries to minimise
            Args:
                wa_vec (np.ndarray): current adjacency vector with intra- and inter-slice weights
            Returns:
                float: objective
            """
            _w_mat, _a_mat = _reshape_wa(wa_vec, d_vars, p_orders)
            loss = (
                0.5 / n
                * np.square(
                    np.linalg.norm(
                        X.dot(np.eye(d_vars, d_vars) - _w_mat) - X_lags.dot(_a_mat), "fro"
                    )
                )
            )
            _h_value = _h(wa_vec)
            l1_penalty = self.lambda_w * (wa_vec[: 2 * d_vars**2].sum()) + self.lambda_a * (
                wa_vec[2 * d_vars**2 :].sum()
            )
            return loss + 0.5 * rho * _h_value * _h_value + alpha * _h_value + l1_penalty

        def _grad(wa_vec: np.ndarray) -> np.ndarray:
            """
            Gradient function used to compute next step in dynotears
            Args:
                wa_vec (np.ndarray): current adjacency vector with intra- and inter-slice weights
            Returns:
                gradient vector
            """
            _w_mat, _a_mat = _reshape_wa(wa_vec, d_vars, p_orders)
            e_mat = slin.expm(_w_mat * _w_mat)
            loss_grad_w = (
                -1.0
                / n
                * (X.T.dot(X.dot(np.eye(d_vars, d_vars) - _w_mat) - X_lags.dot(_a_mat)))
            )
            obj_grad_w = (
                loss_grad_w
                + (rho * (np.trace(e_mat) - d_vars) + alpha) * e_mat.T * _w_mat * 2
            )
            obj_grad_a = (
                -1.0
                / n
                * (X_lags.T.dot(X.dot(np.eye(d_vars, d_vars) - _w_mat) - X_lags.dot(_a_mat)))
            )
            grad_vec_w = np.append(
                obj_grad_w, -obj_grad_w, axis=0
            ).flatten() + self.lambda_w * np.ones(2 * d_vars**2)
            grad_vec_a = obj_grad_a.reshape(p_orders, d_vars**2)
            grad_vec_a = np.hstack(
                (grad_vec_a, -grad_vec_a)
            ).flatten() + self.lambda_a * np.ones(2 * p_orders * d_vars**2)
            return np.append(grad_vec_w, grad_vec_a, axis=0)
        
        for n_iter in range(self.max_iter):
            while (rho < 100) and (h_new > 0.5 * h_value or h_new == np.inf):
                wa_new = sopt.minimize(
                _func, wa_est, method="L-BFGS-B", jac=_grad).x
                h_new = _h(wa_new)
                if h_new > 0.25 * h_value:
                    rho *= 10

            wa_est = wa_new
            h_value = h_new
            alpha += rho * h_value
            if h_value <= self.h_tol:
                break
            if h_value > self.h_tol and n_iter == self.max_iter - 1:
                warnings.warn("Failed to converge. Consider increasing max_iter.")
        W_est, A_est = _reshape_wa(wa_est, d_vars, p_orders)
        W_est[np.abs(W_est) < self.w_threshold] = 0
        W_est[np.abs(W_est) >= self.w_threshold] = 1
        self.DAG = W_est