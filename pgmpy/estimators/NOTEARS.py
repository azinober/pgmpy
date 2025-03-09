import math

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import torch
from scipy.special import expit as sigmoid
from torch.optim import LBFGS
from tqdm.auto import trange

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators import ExpertKnowledge, StructureEstimator
from pgmpy.global_vars import logger
from pgmpy.utils import compat_fns


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

    def __init__(self, data, args=None, **kwargs):
        super(NOTEARS, self).__init__(data=data, **kwargs)
        self.backend = compat_fns.get_compute_backend()
        self.args = args

    def _loss_grad(self, loss_type, data, adjacency_matrix):
        """
        Calculate the value of the loss function and its gradient.

        Parameters
        ----------
        data: numpy.ndarray
            The data passed for DAG estimation.

        adjacency_matrix: numpy.ndarray
            The weighted matrix for the DAG.

        Returns
        -------
        loss: float
            The value of the loss function for given data.

        loss_jac: numpy.ndarray
            The gradient of the loss function for given data.
        """
        if self.backend == torch:
            adjacency_matrix = adjacency_matrix.double()

        M = data @ adjacency_matrix
        if loss_type == "l2":
            R = data - M
            loss = 0.5 / data.shape[0] * (R**2).sum()
            loss_jac = -1.0 / data.shape[0] * data.T @ R

        elif loss_type == "logistic":
            loss = 1.0 / data.shape[0] * (self.backend.logaddexp(0, M) - data * M).sum()
            loss_jac = 1.0 / data.shape[0] * data.T @ (sigmoid(M) - data)

        elif loss_type == "poisson":
            S = self.backend.exp(M)
            loss = 1.0 / data.shape[0] * (S - data * M).sum()
            loss_jac = 1.0 / data.shape[0] * data.T @ (S - data)

        else:
            raise ValueError("unknown loss type")

        return loss, loss_jac

    def _constraint_grad(self, adjacency_matrix):
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
        # if config.get_backend()=="numpy":
        E = slin.expm(adjacency_matrix * adjacency_matrix)
        if config.get_backend() == "torch":
            E = torch.from_numpy(E)
        acyclic_penalty = self.backend.trace(E) - adjacency_matrix.shape[0]
        acyclic_jac = E.T * adjacency_matrix * 2
        return acyclic_penalty, acyclic_jac

    def _adj(self, adjacency_matrix_doubled):
        """
        Convert doubled array ([2 d^2] array) back to original variables ([d, d] matrix).

        Parameters
        ----------

        Returns
        -------
        adjacency_matrix: numpy.ndarray
            The weighted matrix in original dimensions (n*n)
        """
        dim = np.sqrt(adjacency_matrix_doubled.shape[0] / 2).astype(int)
        return (
            adjacency_matrix_doubled[: dim * dim]
            - adjacency_matrix_doubled[dim * dim :]
        ).reshape([dim, dim])

    def _forbidden_penalty_gradient(self, adjacency_matrix, forbidden_mask):
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

        penalty = self.backend.sum(
            (self.backend.abs(adjacency_matrix) * forbidden_mask) ** 2
        )
        jac = 2 * adjacency_matrix * forbidden_mask
        return penalty, jac

    def _fixed_penalty_gradient(
        self, adjacency_matrix, fixed_mask, alpha, weight_threshold
    ):
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
        # changes?

        W_thres = self.backend.where(
            adjacency_matrix < weight_threshold, adjacency_matrix, 0
        )
        penalty_mat = self.backend.exp(
            -((weight_threshold - (W_thres * fixed_mask)) ** (-alpha))
        )
        jac = (
            -alpha
            * fixed_mask
            * penalty_mat
            * (weight_threshold - W_thres * fixed_mask) ** (-alpha - 1)
        )

        return self.backend.sum(penalty_mat), jac

    def _objective_function(
        self,
        adjacency_matrix_doubled,
        alpha,
        rho,
        lambda1,
        lambda2,
        lambda3,
        loss_type,
        data,
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
        adjacency_matrix_est = self._adj(adjacency_matrix_doubled)
        loss, loss_jac = self._loss_grad(loss_type, data, adjacency_matrix_est)
        acyclic_penalty, acyclic_jac = self._constraint_grad(adjacency_matrix_est)

        forb_penalty, forb_jac = self._forbidden_penalty_gradient(
            adjacency_matrix_est, forbidden_mask
        )
        fixed_penalty, fixed_jac = self._fixed_penalty_gradient(
            adjacency_matrix_est, fixed_mask, alpha_2, w_threshold
        )

        obj = (
            loss
            + 0.5 * rho * acyclic_penalty * acyclic_penalty
            + alpha * acyclic_penalty
            + lambda1 * adjacency_matrix_doubled.sum()
            + lambda2 * forb_penalty
            + lambda3 * fixed_penalty
        )
        G_smooth = (
            loss_jac
            + (rho * acyclic_penalty + alpha) * acyclic_jac
            + lambda2 * forb_jac
            + lambda3 * fixed_jac
        )

        if self.backend == torch:
            g_obj = self.backend.concatenate(
                (
                    G_smooth + lambda1 + lambda2 * forb_jac + lambda3 * fixed_jac,
                    -G_smooth + lambda1 + lambda2 * forb_jac + lambda3 * fixed_jac,
                ),
                dim=0,
            )
        else:
            g_obj = self.backend.concatenate(
                (
                    G_smooth + lambda1 + lambda2 * forb_jac + lambda3 * fixed_jac,
                    -G_smooth + lambda1 + lambda2 * forb_jac + lambda3 * fixed_jac,
                ),
                axis=None,
            )
        return obj, g_obj

    def closure(self):
        self.lbfgs.zero_grad()  # Clear previous gradients
        loss, _ = self._objective_function(*self.args)
        loss = torch.tensor(loss, dtype=torch.float32, requires_grad=True)
        loss.backward()
        return loss

    def estimate(
        self,
        lambda1,
        lambda2=10,
        lambda3=10,
        loss_type="l2",
        max_iter=20,
        h_tol=1e-8,
        rho_max=1e16,
        w_threshold=0.3,
        c=0.25,
        alpha_2=10,
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

        expert_knowledge._orient_temporal_forbidden_edges(DAG(), only_edges=False)
        n = self.data.shape[1]
        nodes = self.data.columns.to_list()

        if config.get_backend() == "numpy":
            data = np.array(self.data.values, dtype=config.get_dtype())
        else:
            data = torch.tensor(
                self.data.values, dtype=config.get_dtype(), device=config.get_device()
            )

        # Step 1: Initialize starting estimates
        adjacency_matrix_est, rho, alpha, h = (
            self.backend.zeros(2 * n * n),
            1.0,
            0.0,
            self.backend.inf,
        )  # double w_est into (w_pos, w_neg)

        forbidden_mask = self.backend.zeros((n, n))
        for u, v in expert_knowledge.forbidden_edges:
            i = nodes.index(u)
            j = nodes.index(v)
            forbidden_mask[i][j] = 1

        fixed_mask = self.backend.zeros((n, n))
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
        self.args = (
            adjacency_matrix_est,
            alpha,
            rho,
            lambda1,
            lambda2,
            lambda3,
            loss_type,
            data,
            forbidden_mask,
            fixed_mask,
            w_threshold,
            alpha_2,
        )

        # Why are we normalizing only for l2 - stabilze by reducing value of squared loss
        if loss_type == "l2":
            data = data - self.backend.mean(data, axis=0, keepdims=True)

        # Step 2: Iterate until max_iter iterations
        for _ in iteration:
            adjacency_matrix_new, h_new = None, None

            # Step 2.(a): Solve for new values of W and h
            while rho < rho_max:
                if self.backend == np:
                    sol = sopt.minimize(
                        self._objective_function,
                        adjacency_matrix_est,
                        args=(
                            alpha,
                            rho,
                            lambda1,
                            lambda2,
                            lambda3,
                            loss_type,
                            data,
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
                    h_new, _ = self._constraint_grad(self._adj(adjacency_matrix_new))
                else:
                    self.lbfgs = LBFGS(
                        [adjacency_matrix_est],
                        history_size=10,
                        max_iter=5,
                        line_search_fn="strong_wolfe",
                    )
                    self.lbfgs.step(self.closure)
                    adjacency_matrix_new = adjacency_matrix_est.detach().clone()
                    h_new, _ = self._constraint_grad(self._adj(adjacency_matrix_new))

                # TODO: c is kind of an adaptive step size. Do we need it?
                if h_new > c * h:
                    rho *= 10
                else:
                    break

            # Step 2.(b): Update alpha
            adjacency_matrix_est, h = adjacency_matrix_new, h_new
            alpha += rho * h

            # Step 2.(c): Break away if h converges to 0
            if h <= h_tol or rho >= rho_max:
                break

        # Step 3: Convert doubled matrix to normal n*n and apply threshold to final weighted matrix
        adjacency_matrix_est = self._adj(adjacency_matrix_est)
        adjacency_matrix_est[self.backend.abs(adjacency_matrix_est) < w_threshold] = 0

        edges = []
        for i in range(n):
            for j in range(n):
                if adjacency_matrix_est[i][j] != 0:
                    edges.append((nodes[i], nodes[j]))

        dag = DAG(ebunch=edges)
        dag.add_nodes_from(nodes)
        return dag
