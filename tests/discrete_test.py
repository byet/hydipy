import unittest
from hydipy.Discrete import DiscreteNode
import pyAgrum as gum
import numpy as np


class Test_DiscreteNOde(unittest.TestCase):
    def test_no_parents(self):
        self.discnode = DiscreteNode("a", [[0.7],[0.2],[0.1]], parents=[], states=["a1","a2","a3"])
        self.assertEqual("a", self.discnode.id)
        self.assertIsNone(np.testing.assert_array_equal(np.array([[0.7],[0.2],[0.1]]), self.discnode.values))        
        self.assertListEqual([], self.discnode.parents)
        self.assertListEqual(["a1","a2","a3"], self.discnode.states)
        self.assertEqual(3, self.discnode.cardinality)
        self.assertIsNone(np.testing.assert_array_equal(np.array([0.7,0.2,0.1]),self.discnode.agrum_cpd()))
        bn=gum.BayesNet("test_bn")
        var = self.discnode.agrum_var()
        bn.add(var)
        bn.cpt("a").fillWith(self.discnode.agrum_cpd())
        cpt = bn.cpt("a")
        self.assertTupleEqual(cpt.names, tuple('a'))
        self.assertListEqual(cpt.tolist(), [0.7, 0.2, 0.1])

    def test_one_parent(self):
        self.a = DiscreteNode("a", [[0.7],[0.2],[0.1]], parents=[], states=["a1","a2","a3"])
        self.b = DiscreteNode("b", [[0.4, 0.2, 0.9],[0.6, 0.8, 0.1]], parents=["a"], states=["b1", "b2"])
        self.assertEqual("b", self.b.id)
        self.assertIsNone(np.testing.assert_array_equal(np.array([[0.4, 0.2, 0.9],[0.6, 0.8, 0.1]]), self.b.values))        
        self.assertListEqual(["a"], self.b.parents)
        self.assertListEqual(["b1","b2"], self.b.states)
        self.assertEqual(2, self.b.cardinality)
        self.assertListEqual([],self.a.agrum_edges())
        self.assertListEqual([('a','b')],self.b.agrum_edges())
        self.assertIsNone(np.testing.assert_array_equal(np.array([0.7,0.2,0.1]),self.a.agrum_cpd()))
        self.assertIsNone(np.testing.assert_array_equal(np.array([0.4,0.6,0.2,0.8,0.9,0.1]),self.b.agrum_cpd()))
        bn=gum.BayesNet("test_bn")
        var_a = self.a.agrum_var()
        var_b = self.b.agrum_var()
        bn.add(var_a)
        bn.add(var_b)
        bn.addArcs(self.b.agrum_edges())
        bn.cpt("a").fillWith(self.a.agrum_cpd())
        bn.cpt("b").fillWith(self.b.agrum_cpd())
        self.assertListEqual(bn.cpt("a").tolist(), [0.7, 0.2, 0.1])
        self.assertListEqual(bn.cpt("b").tolist(), [[0.4, 0.6], [0.2, 0.8],[0.9, 0.1]])

    def test_two_parents(self):
        self.a = DiscreteNode("a", [[0.7],[0.2],[0.1]], parents=[], states=["a1","a2","a3"])
        self.b = DiscreteNode("b", [[0.2], [0.8]], parents=[], states=["b1", "b2"])
        self.c = DiscreteNode("c", [[0.4, 0.8, 0.2, 0.9, 0.25, .99],[0.6, 0.2, 0.8, 0.1, 0.75, 0.01]], parents=["a","b"], states=["c1", "c2"])
        self.assertListEqual([('b','c'),('a','c')],self.c.agrum_edges())
        self.assertIsNone(np.testing.assert_array_equal(np.array([0.7,0.2,0.1]),self.a.agrum_cpd()))
        self.assertIsNone(np.testing.assert_array_equal(np.array([0.2,0.8]),self.b.agrum_cpd()))
        self.assertIsNone(np.testing.assert_array_equal(np.array([0.4,0.6,0.8,0.2,0.2,0.8,0.9,0.1,0.25,0.75,0.99,0.01]),self.c.agrum_cpd()))
        bn=gum.BayesNet("test_bn")
        var_a = self.a.agrum_var()
        var_b = self.b.agrum_var()
        var_c = self.c.agrum_var()
        bn.add(var_a)
        bn.add(var_b)
        bn.add(var_c)
        bn.addArcs(self.a.agrum_edges())
        bn.addArcs(self.b.agrum_edges())
        bn.addArcs(self.c.agrum_edges())
        bn.cpt("a").fillWith(self.a.agrum_cpd())
        bn.cpt("b").fillWith(self.b.agrum_cpd())
        bn.cpt("c").fillWith(self.c.agrum_cpd())
        cpt_c = bn.cpt("c")
        self.assertTupleEqual(cpt_c.names, ('c','b','a'))
        self.assertListEqual(cpt_c.tolist(),[[[0.4 , 0.6 ], [0.8 , 0.2 ]], 
                                            [[0.2 , 0.8 ],[0.9 , 0.1 ]],
                                            [[0.25, 0.75],[0.99, 0.01]]])
        self.assertEqual(str(cpt_c), '\n             ||  c                |\nb     |a     ||c1       |c2       |\n------|------||---------|---------|\nb1    |a1    || 0.4000  | 0.6000  |\nb2    |a1    || 0.8000  | 0.2000  |\nb1    |a2    || 0.2000  | 0.8000  |\nb2    |a2    || 0.9000  | 0.1000  |\nb1    |a3    || 0.2500  | 0.7500  |\nb2    |a3    || 0.9900  | 0.0100  |\n')