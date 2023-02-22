import unittest
from hydipy.ParameterLearning import CPTLearner
from hydipy.Discrete import DiscreteNode
from hydipy.DynamicDiscretization import Hybrid_BN
import pandas as pd
import numpy as np


class Test_DiscreteNode(unittest.TestCase):
    def setUp(self):
        self.train_data = pd.DataFrame(
            {"a": ["a1", "a2", "a1"], "b": ["b1", "b1", "b2"], "c": ["c2", "c2", "c1"]},
        )
        a = DiscreteNode(id="a", values=[[0.7], [0.3]], parents=[], states=["a1", "a2"])
        b = DiscreteNode(id="b", values=[[0.4], [0.6]], parents=[], states=["b1", "b2"])
        c = DiscreteNode(
            id="c",
            values=[[0.1, 0.6, 0.3, 0.25], [0.9, 0.4, 0.7, 0.75]],
            parents=["a", "b"],
            states=["c1", "c2"],
        )
        self.model = Hybrid_BN([a, b, c])

    def test_counts_parents(self):

        state_names = {"a": ["a1", "a2"], "b": ["b1", "b2"], "c": ["c1", "c2"]}
        variable = "c"
        parents = ["b", "a"]

        learner = CPTLearner(self.model, self.train_data)
        learner.correction = 0
        cpt_no_correction = learner.cpt_counts(
            variable=variable, state_names=state_names, parents=parents
        )

        col_multi_index = pd.MultiIndex.from_tuples(
            tuples=[("b1", "a1"), ("b1", "a2"), ("b2", "a1"), ("b2", "a2")],
            names=["b", "a"],
        )
        row_index = pd.Index(["c1", "c2"], name="c")
        check_df_no_correction = pd.DataFrame(
            [[0.00, 0.00, 1.00, 0.00], [1.00, 1.00, 0.00, 0.00]],
            index=row_index,
            columns=col_multi_index,
        )

        pd.testing.assert_frame_equal(cpt_no_correction, check_df_no_correction)

        learner = CPTLearner(self.model, self.train_data)
        learner.correction = 0.01
        cpt_correction = learner.cpt_counts(
            variable=variable, state_names=state_names, parents=parents
        )
        check_df_correction = pd.DataFrame(
            [[0.01, 0.01, 1.01, 0.01], [1.01, 1.01, 0.01, 0.01]],
            index=row_index,
            columns=col_multi_index,
        )
        pd.testing.assert_frame_equal(cpt_correction, check_df_correction)

    def test_counts_no_parents(self):

        state_names = {"c": ["c1", "c2"]}
        variable = "c"
        parents = []

        learner = CPTLearner(self.model, self.train_data)
        learner.correction = 0
        cpt_no_correction = learner.cpt_counts(
            variable=variable, state_names=state_names, parents=parents
        )

        row_index = ["c1", "c2"]
        col_index = ["c"]
        check_df_no_correction = pd.DataFrame(
            [[1.0], [2.0]], index=row_index, columns=col_index
        )

        pd.testing.assert_frame_equal(cpt_no_correction, check_df_no_correction)

        learner = CPTLearner(self.model, self.train_data)
        learner.correction = 0.02
        cpt_correction = learner.cpt_counts(
            variable=variable, state_names=state_names, parents=parents
        )

        check_df_correction = pd.DataFrame(
            [[1.02], [2.02]], index=row_index, columns=col_index
        )

        pd.testing.assert_frame_equal(cpt_correction, check_df_correction)

    def test_learn_node_parents(self):
        learner = CPTLearner(self.model, self.train_data)
        node_c = self.model.nodes["c"]
        learned_node_c = learner.learn_cpt(node_c)

        self.assertEqual(type(learned_node_c), DiscreteNode)
        self.assertEqual(learned_node_c.id, "c")
        self.assertEqual(learned_node_c.states, ["c1", "c2"])
        self.assertEqual(learned_node_c.parents, ["a", "b"])
        self.assertIsNone(
            np.testing.assert_array_equal(
                learned_node_c.values,
                np.array([[0.01, 1.01, 0.01, 0.01], [1.01, 0.01, 1.01, 0.01]]),
            )
        )
        learned_node_c.normalize()
        self.assertIsNone(
            np.testing.assert_array_almost_equal(
                learned_node_c.values,
                np.array(
                    [
                        [0.009803922, 0.990196078, 0.009803922, 0.5],
                        [0.990196078, 0.009803922, 0.990196078, 0.5],
                    ]
                ),
            )
        )

        learner = CPTLearner(self.model, self.train_data, correction=0.0)
        learned_node_c2 = learner.learn_cpt(node_c)
        self.assertIsNone(
            np.testing.assert_array_equal(
                learned_node_c2.values,
                np.array([[0.00, 1.00, 0.00, 0.00], [1.00, 0.00, 1.00, 0.00]]),
            )
        )

        learned_node_c2.normalize()
        self.assertIsNone(
            np.testing.assert_array_almost_equal(
                learned_node_c2.values,
                np.array([[0.00, 1.00, 0.00, 0.50], [1.00, 0.00, 1.00, 0.50]]),
            )
        )

    def test_learn_node_no_parents(self):
        learner = CPTLearner(self.model, self.train_data)
        node_a = self.model.nodes["a"]
        learned_node_a = learner.learn_cpt(node_a)

        self.assertEqual(type(learned_node_a), DiscreteNode)
        self.assertEqual(learned_node_a.id, "a")
        self.assertEqual(learned_node_a.states, ["a1", "a2"])
        self.assertEqual(learned_node_a.parents, [])
        self.assertIsNone(
            np.testing.assert_array_equal(
                learned_node_a.values, np.array([[2.01], [1.01]])
            )
        )

        learned_node_a.normalize()
        self.assertIsNone(
            np.testing.assert_array_almost_equal(
                learned_node_a.values, np.array([[0.665562914], [0.334437086]])
            )
        )

        learner = CPTLearner(self.model, self.train_data, correction=0.0)
        learned_node_a2 = learner.learn_cpt(node_a)
        self.assertIsNone(
            np.testing.assert_array_equal(
                learned_node_a2.values, np.array([[2.00], [1.00]])
            )
        )
        learned_node_a2.normalize()
        self.assertIsNone(
            np.testing.assert_array_almost_equal(
                learned_node_a2.values, np.array([[0.75], [0.25]])
            )
        )


    def test_mle_learn(self):
        learner = CPTLearner(self.model, self.train_data)
        learned_cpts = learner.mle()

        a = DiscreteNode(id="a", values=[[0.665562914], [0.334437086]], parents=[], states=["a1", "a2"])
        b = DiscreteNode(id="b", values=[[0.665562914], [0.334437086]], parents=[], states=["b1", "b2"])
        c = DiscreteNode(
            id="c",
            values= [
                        [0.009803922, 0.990196078, 0.009803922, 0.5],
                        [0.990196078, 0.009803922, 0.990196078, 0.5],
                    ],
            parents=["a", "b"],
            states=["c1", "c2"],
        )

        self.assertTrue(a == learned_cpts['a'])
        self.assertTrue(b == learned_cpts['b'])
        self.assertTrue(c == learned_cpts['c'])
