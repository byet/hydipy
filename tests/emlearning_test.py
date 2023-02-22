import unittest
from hydipy.ParameterLearning import EMLearner
from hydipy.Discrete import DiscreteNode
from hydipy.DynamicDiscretization import Hybrid_BN
import pandas as pd
import numpy as np
import pyAgrum as gum

class Test_DiscreteNode(unittest.TestCase):
    def setUp(self):
        self.train_data = pd.DataFrame(
            {"a": ["a1", "a2", "a1"], "b": ["b1", pd.NA, "b2"], "c": ["c2", "c2", pd.NA]},
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
    
    def test_em_object_init(self):
        eml = EMLearner(self.model, self.train_data)
        expected = pd.DataFrame(
            {"a": ["a2", "a1"], "b": [pd.NA, "b2"], "c": ["c2", pd.NA]}, index=[1,2])
        pd.testing.assert_frame_equal(eml.missing_data, expected)
        self.assertEqual(eml.nrows_missing, 2)

    def test_missing_variables(self):
        eml = EMLearner(self.model, self.train_data)
        self.assertDictEqual(eml.missing_variables(), {'b':['b'],'c':['c','a','b']})

    def test_init_full_counts(self):
        eml = EMLearner(self.model, self.train_data)
        expected = {'a':np.array([[2.01],[1.01]]),'b':np.array([[1.01],[1.01]]),'c':np.array([[0.01,0.01,0.01,0.01],[1.01,0.01,0.01,0.01]])}
        outcome = eml.init_full_counts()
        self.assertIsNone(np.testing.assert_array_equal(expected['a'], outcome['a']))
        self.assertIsNone(np.testing.assert_array_equal(expected['b'], outcome['b']))
        self.assertIsNone(np.testing.assert_array_equal(expected['c'], outcome['c']))
        self.assertTrue(expected.keys() == outcome.keys())

    def test_get_posterior(self):
        eml = EMLearner(self.model, self.train_data)
        missing_variables = eml.missing_variables()
        vars_in_cpt_c = missing_variables['c']
        ie = gum.LazyPropagation(self.model.aux_bn)
        for key, nodes in missing_variables.items():
            ie.addJointTarget(set(nodes))
        outcome = eml.get_posterior(ie, vars_in_cpt_c)
        expected = np.array([[0.028, 0.252,	0.036,	0.045],[0.252,	0.168,	0.084, 0.135]])
        self.assertIsNone(np.testing.assert_array_almost_equal(expected, outcome))

    def test_define_cpt(self):
        eml = EMLearner(self.model, self.train_data)
        node_c = self.model.nodes['c']
        counts = eml.init_full_counts()
        outcome = eml.define_cpt(node_c, counts['c'])

        expected = DiscreteNode(
            id="c",
            values= [
                        [0.009803922, 0.5, 0.5, 0.5],
                        [0.990196078, 0.5, 0.5, 0.5],
                    ],
            parents=["a", "b"],
            states=["c1", "c2"],
        )

        self.assertTrue(outcome == expected)

    def test_e_step(self):
        eml = EMLearner(self.model, self.train_data)
        counts = eml.init_full_counts()
        outcome = eml.e_step(counts)
        expected = {'a':np.array([[2.01],[1.01]]),'b':np.array([[1.393561644],[1.626438356]]),
        'c':np.array([[0.01,0.61,0.01,0.01],[1.01,0.41,0.393561644,0.626438356]])}
        self.assertIsNone(np.testing.assert_array_almost_equal(expected['a'], outcome['a']))
        self.assertIsNone(np.testing.assert_array_almost_equal(expected['b'], outcome['b']))
        self.assertIsNone(np.testing.assert_array_almost_equal(expected['c'], outcome['c']))
        self.assertTrue(expected.keys() == outcome.keys())


    def test_m_step(self):
        eml = EMLearner(self.model, self.train_data)
        counts = eml.init_full_counts()
        e_counts = eml.e_step(counts)
        outcome = eml.m_step(e_counts)
     
        a = DiscreteNode(id="a", values=[[0.665562914], [0.334437086]], parents=[], states=["a1", "a2"])
        b = DiscreteNode(id="b", values=[[0.461444253], [0.538555747]], parents=[], states=["b1", "b2"])
        c = DiscreteNode(
            id="c",
            values= [
                        [0.009803922, 0.598039216, 0.024779362, 0.015712441],
                        [0.990196078, 0.401960784, 0.975220638, 0.984287559],
                    ],
            parents=["a", "b"],
            states=["c1", "c2"],
        )

        self.assertTrue(a == outcome['a'])
        self.assertTrue(b == outcome['b'])
        self.assertTrue(c == outcome['c'])

    def test_em(self):
        eml = EMLearner(self.model, self.train_data)
        outcome = eml.em(5)

        a = DiscreteNode(id="a", values=[[0.665562914], [0.334437086]], parents=[], states=["a1", "a2"])
        b = DiscreteNode(id="b", values=[[0.499419889], [0.500580111]], parents=[], states=["b1", "b2"])
        c = DiscreteNode(
            id="c",
            values= [
                        [0.009803922, 0.590573081, 0.019295779, 0.019166196],
                        [0.990196078, 0.409426919, 0.980704221, 0.980833804],
                    ],
            parents=["a", "b"],
            states=["c1", "c2"],
        )
        self.assertTrue(a == outcome['a'])
        self.assertTrue(b == outcome['b'])
        self.assertTrue(c == outcome['c'])
