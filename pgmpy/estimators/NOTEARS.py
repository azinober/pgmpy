import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators import StructureEstimator
from pgmpy.global_vars import logger


class NOTEARS(StructureEstimator):
    """
    Implementation of the NOTEARS structure learning algorithm.

    NOTEARS is a structure learning algorithm that works by converting the discrete score-based
    DAG learning problem into a continuous optimisation problem.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    References
    ----------
    Xun Zheng, Bryon Aragam, Pradeep Ravikumar, and Eric P. Xing. 2018. DAGs with NO TEARS: continuous
    optimization for structure learning. In Proceedings of the 32nd International Conference on Neural
    Information Processing Systems (NIPS'18). Curran Associates Inc., Red Hook, NY, USA, 9492–9503.
    """

    def __init__(self, data, use_cache=True, **kwargs):
        self.use_cache = use_cache

        super(NOTEARS, self).__init__(data=data, **kwargs)

    def _loss_function(self, loss_type, X, W):
        """
        Calculate the value of the loss function and its gradient.
        """
        M = X @ W
        if loss_type == "l2":
            R = X - M
            loss = 0.5 / X.shape[0] * (R**2).sum()
            G_loss = -1.0 / X.shape[0] * X.T @ R

        elif loss_type == "logistic":
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)

        elif loss_type == "poisson":
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)

        else:
            raise ValueError("unknown loss type")

        return loss, G_loss

    def _constraint_grad(self, W, d):
        """
        Calculate the value of the acyclicity constraint and its gradient.

        The acyclicity constraint is given by  -
            h(W) = trace(exp(W*W)) - d
                where W is the weighted matrix and d is the number of nodes in graph
        """
        E = slin.expm(W * W)
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(self, w, d):
        """
        Convert doubled array ([2 d^2] array) back to original variables ([d, d] matrix).
        """
        return (w[: d * d] - w[d * d :]).reshape([d, d])

    def _objective_function(self, w, alpha, rho, lambda1, d, loss_type, X):
        """
        Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array).
        """
        W = self._adj(w, d)
        loss, G_loss = self._loss_function(loss_type, X, W)
        h, G_h = self._constraint_grad(W, d)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, -G_smooth + lambda1), axis=None)
        return obj, g_obj

    def estimate(
        self,
        lambda1,
        loss_type="l2",
        max_iter=100,
        h_tol=1e-8,
        rho_max=1e16,
        w_threshold=0.3,
    ):
        """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

        The objective function, consisiting of the sum of a loss function and the L1 norm,
        is minimised using the augmented Lagrangian method. The resulting weighted matrix
        is the Structural Equation Model corresponding to the data. The weighted matrix is
        converted to a DAG object and returned.

        Parameters
        ----------
        lambda1: float
            penalty parameter for l1 norm

        loss_type: str (one of "l2", "logistic", "poisson")
            the loss function to be used in objective function

        max_iter: int
            maximum number of dual ascent steps

        h_tol: float
            exit if |h(w_est)| <= htol

        rho_max: float
            exit if rho >= rho_max

        w_threshold: float
            drop edge if |weight| < w_threshold

        Returns
        -------
        Estimated model: pgmpy.base.DAG
            The estimated model with (local) minimum value of objective function.

        Examples
        --------
        """
        # num_samples = self.data.shape[0]
        n = self.data.shape[1]
        nodes = self.data.columns.to_list()
        X = self.data.to_numpy()
        w_est, rho, alpha, h = (
            np.zeros(2 * n * n),
            1.0,
            0.0,
            np.inf,
        )  # double w_est into (w_pos, w_neg)

        bounds = [
            (0, 0) if i == j else (0, None)
            for _ in range(2)
            for i in range(n)
            for j in range(n)
        ]
        if loss_type == "l2":
            X = X - np.mean(X, axis=0, keepdims=True)

        for _ in range(max_iter):
            w_new, h_new = None, None
            while rho < rho_max:
                sol = sopt.minimize(
                    self._objective_function,
                    w_est,
                    args=(alpha, rho, lambda1, n, loss_type, X),
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                )
                w_new = sol.x
                h_new, _ = self._constraint_grad(self._adj(w_new, n), n)
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            w_est, h = w_new, h_new
            alpha += rho * h
            if h <= h_tol or rho >= rho_max:
                break

        W_est = self._adj(w_est, n)
        W_est[np.abs(W_est) < w_threshold] = 0

        edges = []
        for i in range(n):
            for j in range(n):
                if W_est[i][j] != 0:
                    edges.append((nodes[i], nodes[j]))

        return DAG(ebunch=edges)
