import math

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from tqdm.auto import trange

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators import ExpertKnowledge, StructureEstimator
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

    def _loss_grad(self, loss_type, df, adjacency_matrix):
        """
        Calculate the value of the loss function and its gradient.

        Parameters
        ----------
        df: numpy.ndarray
            The data passed for DAG estimation.

        adjacency_matrix: numpy.ndarray
            The weighted matrix for the DAG.

        Returns
        -------
        loss: float
            The value of the loss function for given data.

        G_loss: numpy.ndarray
            The gradient of the loss function for given data.
        """
        M = df @ adjacency_matrix
        if loss_type == "l2":
            R = df - M
            loss = 0.5 / df.shape[0] * (R**2).sum()
            G_loss = -1.0 / df.shape[0] * df.T @ R

        elif loss_type == "logistic":
            loss = 1.0 / df.shape[0] * (np.logaddexp(0, M) - df * M).sum()
            G_loss = 1.0 / df.shape[0] * df.T @ (sigmoid(M) - df)

        elif loss_type == "poisson":
            S = np.exp(M)
            loss = 1.0 / df.shape[0] * (S - df * M).sum()
            G_loss = 1.0 / df.shape[0] * df.T @ (S - df)

        else:
            raise ValueError("unknown loss type")

        return loss, G_loss

    def _constraint_grad(self, adjacency_matrix, n):
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
        E = slin.expm(adjacency_matrix * adjacency_matrix)
        h = np.trace(E) - n
        G_h = E.T * adjacency_matrix * 2
        return h, G_h

    def _adj(self, adjacency_matrix_doubled, n):
        """
        Convert doubled array ([2 d^2] array) back to original variables ([d, d] matrix).

        Parameters
        ----------

        Returns
        -------
        adjacency_matrix: numpy.ndarray
            The weighted matrix in original dimensions (n*n)
        """
        return (
            adjacency_matrix_doubled[: n * n] - adjacency_matrix_doubled[n * n :]
        ).reshape([n, n])

    def _forbidden_penalty_gradient(self, W, forbidden_mask):
        """
        Evaluate value and gradient for masking adjacency matrix using forbidden edge data.

        Parameters
        ----------

        Returns
        -------
        loss: float
            The value of the objective function for given parameters.

        grad: numpy.ndarray
            The gradient of the objective function for given parameters.
        """
        # Given a mask M, and adjacency matrix W, the penalty is given by $ \sum_{i,j} M_{ij} W_{ij}^2 $.
        # The Jacobian is given by $ J_{ij} = 2 * W_{ij} M_{ij} M_{ij} = 2 * W_{ij} * M_{ij} $.

        penalty = np.sum((np.abs(W) * forbidden_mask) ** 2)
        jac = 2 * W * forbidden_mask
        return penalty, jac

    def _fixed_penalty_gradient(self, W, fixed_mask, alpha, w_threshold):
        """
        Evaluate value and gradient for masking adjacency matrix using required edge data.

        Parameters
        ----------

        Returns
        -------
        loss: float
            The value of the objective function for given parameters.

        grad: numpy.ndarray
            The gradient of the objective function for given parameters.
        """
        # Given a mask M, and adjacency matrix W, we define W^t = W if W < w_threshold else 0.
        # The penalty matrix is: $ P_{ij} e^{-(w_t - W^t_{ij} * M_{ij})^{- \alpha}} $
        # The penalty is $ \sum_{ij} P_{ij} $.
        # The Jacobian is $ J_{ij} = -\alpha M_{ij} P_{ij} (w_t -  W_{ij} M_{ij})^{- \alpha - 1} $.

        W_thres = np.where(W < w_threshold, W, 0)
        penalty_mat = np.exp(-(((w_threshold - W_thres) * fixed_mask) ** (-alpha)))
        jac = (
            -alpha
            * fixed_mask
            * penalty_mat
            * (w_threshold - W_thres * fixed_mask) ** (-alpha - 1)
        )

        return np.sum(penalty_mat), jac

    def _objective_function(
        self,
        adjacency_matrix_doubled,
        alpha,
        rho,
        lambda1,
        lambda2,
        lambda3,
        n,
        loss_type,
        df,
        forbidden_mask,
        fixed_mask,
        w_threshold,
        alpha_2,
    ):
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
        adjacency_matrix_est = self._adj(adjacency_matrix_doubled, n)
        loss, G_loss = self._loss_grad(loss_type, df, adjacency_matrix_est)
        h, G_h = self._constraint_grad(adjacency_matrix_est, n)

        forb_penalty, forb_jac = self._forbidden_penalty_gradient(
            adjacency_matrix_est, forbidden_mask
        )
        fixed_penalty, fixed_jac = self._fixed_penalty_gradient(
            adjacency_matrix_est, fixed_mask, alpha_2, w_threshold
        )
        obj = (
            loss
            + 0.5 * rho * h * h
            + alpha * h
            + lambda1 * adjacency_matrix_doubled.sum()
            + lambda2 * forb_penalty
            + lambda3 * fixed_penalty
        )
        G_smooth = (
            G_loss + (rho * h + alpha) * G_h + lambda2 * forb_jac + lambda3 * fixed_jac
        )

        # print(G_smooth, lambda1)
        g_obj = np.concatenate(
            (
                G_smooth + lambda1 + lambda2 * forb_jac + lambda3 * fixed_jac,
                -G_smooth + lambda1 + lambda2 * forb_jac + lambda3 * fixed_jac,
            ),
            axis=None,
        )
        return obj, g_obj

    def estimate(
        self,
        lambda1,
        loss_type="l2",
        max_iter=20,
        h_tol=1e-8,
        rho_max=1e16,
        w_threshold=0.1,
        c=0.25,
        alpha_2=1,
        expert_knowledge=None,
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

        if expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()

        n = self.data.shape[1]
        nodes = self.data.columns.to_list()

        # TODO: For utilizing the torch backend we would need to define all the np. functions using backend.
        # TODO: This needs to be changed to also work with pytorch tensors
        df = self.data.to_numpy()

        # Step 1: Initialize starting estimates
        adjacency_matrix_est, rho, alpha, h = (
            np.zeros(2 * n * n),
            1.0,
            0.0,
            np.inf,
        )  # double w_est into (w_pos, w_neg)
        lambda2, lambda3 = 1, 1

        forbidden_mask = np.zeros(shape=(n, n))
        for u, v in expert_knowledge.forbidden_edges:
            i = nodes.index(u)
            j = nodes.index(v)
            forbidden_mask[i][j] = 1
        # print(forbidden_mask)
        # print(nodes)
        fixed_mask = np.zeros(shape=(n, n))
        for u, v in expert_knowledge.required_edges:
            i = nodes.index(u)
            j = nodes.index(v)
            fixed_mask[i][j] = 1

        bounds = [
            (0, 0) if i == j else (0, None)
            for _ in range(2)
            for i in range(n)
            for j in range(n)
        ]

        # Why are we normalizing only for l2
        if loss_type == "l2":
            df = df - np.mean(df, axis=0, keepdims=True)

        # Step 2: Iterate until max_iter iterations
        for _ in iteration:
            adjacency_matrix_new, h_new = None, None

            # Step 2.(a): Solve for new values of W and h
            while rho < rho_max:
                sol = sopt.minimize(
                    self._objective_function,
                    adjacency_matrix_est,
                    args=(
                        alpha,
                        rho,
                        lambda1,
                        lambda2,
                        lambda3,
                        n,
                        loss_type,
                        df,
                        forbidden_mask,
                        fixed_mask,
                        w_threshold,
                        alpha_2,
                    ),
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                )
                adjacency_matrix_new = sol.x
                h_new, _ = self._constraint_grad(self._adj(adjacency_matrix_new, n), n)

                # TODO: c is kind of an adaptive step size. Do we need it?
                if h_new > c * h:
                    rho *= 10
                    lambda2 *= 2
                    lambda3 *= 1.1
                else:
                    break

            # Step 2.(b): Update alpha
            adjacency_matrix_est, h = adjacency_matrix_new, h_new
            alpha += rho * h

            # Step 2.(c): Break away if h converges to 0
            if h <= h_tol or rho >= rho_max:
                # print(rho,rho_max, h, h_tol)
                break

        # Step 3: Convert doubled matrix to normal n*n and apply threshold to final weighted matrix
        adjacency_matrix_est = self._adj(adjacency_matrix_est, n)
        adjacency_matrix_est[np.abs(adjacency_matrix_est) < w_threshold] = 0

        edges = []
        for i in range(n):
            for j in range(n):
                if adjacency_matrix_est[i][j] != 0:
                    edges.append((nodes[i], nodes[j]))

        return DAG(ebunch=edges)
