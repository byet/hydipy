import unittest
from hydipy import Continuous
from hydipy.Distributions import NormalDistribution
import numpy as np

class Test_Discretization(unittest.TestCase):
    def setUp(self):
        self.contnode = Continuous.DiscretizedNode("testnode")
        
    def test_set_discretization(self):
        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.disc, np.array([-1,-0.5,0.5,1])))
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.lb, np.array([-1,-0.5,0.5])))
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.ub, np.array([-0.5,0.5,1])))
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.median, np.array([-0.75,0.,0.75])))
        self.assertListEqual(self.contnode.states, ['-1.0,-0.5', '-0.5,0.5', '0.5,1.0'])
        self.assertEqual(self.contnode.cardinality, 3)


    def test_discretized_pmb(self):
        dist = NormalDistribution(0.5, 1.1)
        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        probs = self.contnode.discretized_probability_mass(dist, self.contnode.lb, self.contnode.ub)
        self.assertIsNone(np.testing.assert_array_almost_equal(probs, np.array([0.1618330, 0.5405448, 0.2976222])))

    def test_compute_error(self):
        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        probs = np.array([0.1618330, 0.5405448, 0.2976222])
        entropy_error = np.array([0.09794874, 0.05435651, 0.09219452])
        error_computed = self.contnode.compute_error(probs, self.contnode.lb, self.contnode.ub)
        self.assertIsNone(np.testing.assert_array_almost_equal(error_computed, entropy_error))

    def test_set_evidence(self):
        evidence_threshold = 0.01
        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        value = 0.75
        self.contnode.set_evidence(value, evidence_threshold)
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.disc, np.array([-1,-0.5,0.5,0.7425,0.7575,1])))

        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        self.contnode.set_evidence(0.5, evidence_threshold)
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.disc, np.array([-1,-0.5,0.495,0.505,1])))

        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        self.contnode.set_evidence(-1, evidence_threshold)
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.disc, np.array([-1.01,-0.99,-0.5,0.5,1])))

        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        self.contnode.set_evidence(1, evidence_threshold)
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.disc, np.array([-1,-0.5,0.5,0.99,1.01])))

        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        self.contnode.set_evidence(2, evidence_threshold)
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.disc, np.array([-1,-0.5,0.5,1,1.98,2.02])))

        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        self.contnode.set_evidence(-2, evidence_threshold)
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.disc, np.array([-2.02,-1.98,-1,-0.5,0.5,1])))
    

    def test_add_interval(self):
        new_disc = self.contnode.add_interval(np.array([-1,-0.5,0.5,1]), index = 1)
        self.assertIsNone(np.testing.assert_array_equal(new_disc, np.array([-1,-0.5,0.,0.5,1])))

        new_disc = self.contnode.add_interval(np.array([-1,-0.5,0.5,1]), index = 2)
        self.assertIsNone(np.testing.assert_array_equal(new_disc, np.array([-1,-0.5,0.5,0.75,1])))


    def test_merge_interval(self):
        new_disc = self.contnode.merge_interval(np.array([-1,-0.5,0.5,1]), index = 1)
        self.assertIsNone(np.testing.assert_array_equal(new_disc, np.array([-1,0.5,1])))

        new_disc = self.contnode.merge_interval(np.array([-1,-0.5,0.5,1]), index = 2)
        self.assertIsNone(np.testing.assert_array_equal(new_disc, np.array([-1,-0.5,1])))

        new_disc = self.contnode.merge_interval(np.array([-1,-0.5,0.5,1]), index = 3)
        self.assertIsNone(np.testing.assert_array_equal(new_disc, np.array([-1,-0.5,0.5])))

    def test_update_intervals(self):
        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        probs = np.array([0.1618330, 0.5405448, 0.2976222])
        new_disc, added_index, removed_indices = self.contnode.update_intervals(probs)
        self.assertIsNone(np.testing.assert_array_equal(new_disc, np.array([-1,-0.75,-0.5,0.5,1])))
        self.assertEqual(added_index, 0)
        self.assertIsNone(np.testing.assert_array_equal(removed_indices, np.array([])))

        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        probs = np.array([0.25, 0.55, 0.25])
        new_disc, added_index, removed_indices = self.contnode.update_intervals(probs)
        self.assertIsNone(np.testing.assert_array_equal(new_disc, np.array([-1,-0.75,-0.5,0.5,1])))
        self.assertEqual(added_index, 0)
        self.assertIsNone(np.testing.assert_array_equal(removed_indices, np.array([])))

        
        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1,5,10]))
        probs = np.array([0.1618330, 0.5405448, 0.2976221, 0.01, 0])
        error = self.contnode.compute_error(probs, self.contnode.lb, self.contnode.ub)
        new_disc, added_index, removed_indices = self.contnode.update_intervals(probs)
        self.assertIsNone(np.testing.assert_array_equal(new_disc, self.contnode.disc))
        self.assertIsNone(np.testing.assert_array_equal(new_disc, np.array([-1,-0.5,0.5,1,3,5,10])))
        self.assertEqual(added_index, 3, msg=f"{error}")
        self.assertIsNone(np.testing.assert_array_equal(removed_indices, np.array([])))

if __name__ == '__main__':
    unittest.main()