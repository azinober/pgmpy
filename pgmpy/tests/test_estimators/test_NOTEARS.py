import unittest

import numpy as np
import pandas as pd

from pgmpy.estimators import NOTEARS
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

    def test_estimate_ecoli(self):
        est_edges = self.est_ecoli.estimate(
            lambda1=0.5, max_iter=50, h_tol=1e-6, c=0.5, show_progress=False
        ).edges()
        self.assertTrue(len(set(est_edges) - self.ecoli_edges) <= 10)

    def tearDown(self):
        del self.rand_data
        del self.est_rand_data
        del self.model1
        del self.titanic_data
        del self.titanic_data1
        del self.est_titanic1
