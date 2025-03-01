import unittest

import numpy as np
import pandas as pd

from pgmpy import config
from pgmpy.estimators import NOTEARS, ExpertKnowledge
from pgmpy.models import BayesianNetwork
from pgmpy.utils import get_example_model


class TestNoTEARS(unittest.TestCase):
    def setUp(self):
        self.rand_data = pd.DataFrame(
            np.random.randint(0, 4, size=(1000, 3)), columns=list("ABD")
        )
        self.rand_data["C"] = self.rand_data["A"] - self.rand_data["B"]
        self.rand_data["D"] += self.rand_data["A"]
        self.est_rand_data = NOTEARS(self.rand_data)

        self.model1 = BayesianNetwork()
        self.model1.add_nodes_from(["A", "B", "C"])
        self.model1.add_edge("A", "B")
        self.model1_possible_edges = set(
            [(u, v) for u in self.model1.nodes() for v in self.model1.nodes()]
        )

        self.ecoli_data = get_example_model("ecoli70").simulate(int(3e4), seed=42)
        self.ecoli_edges = set(get_example_model("ecoli70").edges())
        self.est_ecoli = NOTEARS(self.ecoli_data)

    def test_estimate_rand(self):
        dag_rand_data = self.est_rand_data.estimate(lambda1=0.01, show_progress=False)
        self.assertSetEqual(
            set(dag_rand_data.edges()), set([("A", "D"), ("A", "C"), ("B", "C")])
        )

        dag_rand_logistic = self.est_rand_data.estimate(
            lambda1=0.01, loss_type="logistic", show_progress=False
        )
        self.assertSetEqual(
            set(dag_rand_logistic.edges()), set([("A", "D"), ("A", "C"), ("B", "C")])
        )

    def test_estimate_expert(self):
        self.rand_data["E"] = self.rand_data["D"] + self.rand_data["B"]
        expert_knowledge = ExpertKnowledge(forbidden_edges=[("D", "E")])
        dag_rand_data = NOTEARS(self.rand_data).estimate(
            lambda1=0.4,
            alpha_2=5,
            rho_max=1e20,
            expert_knowledge=expert_knowledge,
            max_iter=100,
            show_progress=False,
        )
        self.assertSetEqual(
            set(dag_rand_data.edges()),
            set(
                [("A", "D"), ("A", "C"), ("B", "C"), ("B", "E")]
            ),  # Should not contain ("D","E")
        )

    def test_estimate_ecoli(self):
        est_edges = self.est_ecoli.estimate(
            lambda1=0.4, max_iter=5, h_tol=1e-6, c=0.5, show_progress=False
        ).edges()
        self.assertTrue(len(set(est_edges) - self.ecoli_edges) <= 10)

    def tearDown(self):
        del self.rand_data
        del self.est_rand_data
        del self.model1
        del self.ecoli_data
        del self.ecoli_edges
        del self.est_ecoli


class TestNoTEARSTorch(TestNoTEARS):
    def setUp(self):
        config.set_backend("torch")
        super().setUp()

    def tearDown(self):
        super().tearDown()
        config.set_backend("numpy")
