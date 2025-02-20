import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from tqdm.auto import trange

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

    def __init__(self, data, **kwargs):
        super(NOTEARS, self).__init__(data=data, **kwargs)

    def _loss_grad(self, loss_type, X, W):
        """
        Calculate the value of the loss function and its gradient.

        Parameters
        ----------
        X: numpy.ndarray
            The data used for DAG estimation.

        W: numpy.ndarray
            The weighted matrix for the DAG.

        Returns
        -------
        loss: float
            The value of the loss function for given data.

        G_loss: numpy.ndarray
            The gradient of the loss function for given data.
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

    def _constraint_grad(self, W, n):
        """
        Calculate the value of the acyclicity constraint and its gradient.

        The acyclicity constraint is given by  -
            h(W) = trace(exp(W*W)) - d
                where W is the weighted matrix and d is the number of nodes in graph.

        Parameters
        ----------

        Returns
        -------
        h: float
            The value of the constraint function for given parameters.

        g_h: numpy.ndarray
            The gradient of the objective function for given parameters.
        """
        E = slin.expm(W * W)
        h = np.trace(E) - n
        G_h = E.T * W * 2
        return h, G_h

    def _adj(self, w, n):
        """
        Convert doubled array ([2 d^2] array) back to original variables ([d, d] matrix).

        Parameters
        ----------

        Returns
        -------
        w: numpy.ndarray
            The weighted matrix in original dimensions (n*n)
        """
        return (w[: n * n] - w[n * n :]).reshape([n, n])

    def _objective_function(self, w, alpha, rho, lambda1, n, loss_type, X):
        """
        Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array).

        Parameters
        ----------

        Returns
        -------
        obj: float
            The value of the objective function for given parameters.

        g_obj: numpy.ndarray
            The gradient of the objective function for given parameters.
        """
        W = self._adj(w, n)
        loss, G_loss = self._loss_grad(loss_type, X, W)
        h, G_h = self._constraint_grad(W, n)

        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, -G_smooth + lambda1), axis=None)
        return obj, g_obj

    def estimate(
        self,
        lambda1,
        loss_type="l2",
        max_iter=20,
        h_tol=1e-8,
        rho_max=1e16,
        w_threshold=0.3,
        c=0.25,
        show_progress=True,
    ):
        """
        Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

        The objective function, consisiting of the sum of a loss function and the L1 norm,
        is minimised using the augmented Lagrangian method. The resulting weighted matrix
        is the Structural Equation Model corresponding to the data. The weighted matrix W is
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

        c: float
            progress rate in the range (0, 1)

        Returns
        -------
        Estimated model: pgmpy.base.DAG
            The estimated model with (local) minimum value of objective function.

        Examples
        --------
        """

        # Step 0: Initial checks and setup
        if loss_type not in ("l2", "logistic", "poisson"):
            raise ValueError(
                f"loss_type must be one of: l2, logistic, or poisson. Got: {loss_type}"
            )
        if c <= 0 or c >= 1:
            raise ValueError("c (progress rate) must be in the range (0, 1). Aborting")

        if show_progress and config.SHOW_PROGRESS:
            iteration = trange(int(max_iter))
        else:
            iteration = range(int(max_iter))

        n = self.data.shape[1]
        nodes = self.data.columns.to_list()
        X = self.data.to_numpy()

        # Step 1: Initialize starting guesses
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

        # Step 2: Iterate until max_iter iterations
        for _ in iteration:
            w_new, h_new = None, None

            # Step 2.(a): Solve for new values of W and h
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
                if h_new > c * h:
                    rho *= 10
                else:
                    break

            # Step 2.(b): Update alpha
            w_est, h = w_new, h_new
            alpha += rho * h

            # Step 2.(c): Break away if h converges to 0
            if h <= h_tol or rho >= rho_max:
                break

        # Step 3: Apply threshold to final weighted matrix
        W_est = self._adj(w_est, n)
        W_est[np.abs(W_est) < w_threshold] = 0

        edges = []
        for i in range(n):
            for j in range(n):
                if W_est[i][j] != 0:
                    edges.append((nodes[i], nodes[j]))

        return DAG(ebunch=edges)
