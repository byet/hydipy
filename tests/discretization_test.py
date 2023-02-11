import unittest
from dydipy import ContinuousNodes
from dydipy.Distributions import NormalDistribution
import numpy as np

class Test_Discretization(unittest.TestCase):
    def setUp(self):
        self.contnode = ContinuousNodes.DiscretizedNode()
        
    def test_set_discretization(self):
        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.disc, np.array([-1,-0.5,0.5,1])))
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.lb, np.array([-1,-0.5,0.5])))
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.ub, np.array([-0.5,0.5,1])))
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.median, np.array([-0.75,0.,0.75])))

    def test_build_prob(self):
        dist = NormalDistribution(0.5, 1.1)
        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        intervals, probs = self.contnode.build_prob(dist)
        self.assertIsNone(np.testing.assert_array_equal(intervals, np.array([[-1., -0.5],[-0.5, 0.5],[0.5, 1.]])))
        self.assertIsNone(np.testing.assert_array_almost_equal(probs, np.array([0.1618330, 0.5405448, 0.2976222])))

    def test_compute_error(self):
        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        probs = np.array([0.1618330, 0.5405448, 0.2976222])
        entropy_error = np.array([0.09794874, 0.05435651, 0.09219452])
        error_computed = self.contnode.compute_error(probs)
        self.assertIsNone(np.testing.assert_array_almost_equal(error_computed, entropy_error))

    def test_set_evidence(self):
        evidence_threshold = 0.01
        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        self.contnode.set_evidence(0.75, evidence_threshold)
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.disc, np.array([-1,-0.5,0.5,0.74,0.76,1])))

        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        self.contnode.set_evidence(0.5, evidence_threshold)
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.disc, np.array([-1,-0.5,0.49,0.51,1])))

        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        self.contnode.set_evidence(-1, evidence_threshold)
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.disc, np.array([-1.01,-0.99,-0.5,0.5,1])))

        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        self.contnode.set_evidence(1, evidence_threshold)
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.disc, np.array([-1,-0.5,0.5,0.99,1.01])))

        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        self.contnode.set_evidence(2, evidence_threshold)
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.disc, np.array([-1,-0.5,0.5,1,1.99,2.01])))

        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        self.contnode.set_evidence(-2, evidence_threshold)
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.disc, np.array([-2.01,-1.99,-1,-0.5,0.5,1])))
    

    def test_add_interval(self):
        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        self.contnode.add_interval(index = 1)
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.disc, np.array([-1,-0.5,0.,0.5,1])))

        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        self.contnode.add_interval(index = 2)
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.disc, np.array([-1,-0.5,0.5,0.75,1])))

    def test_add_interval_index_error(self):
        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        self.assertRaises(IndexError, self.contnode.add_interval(index = 3))

    def test_merge_interval(self):
        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        self.contnode.merge_interval(index = 1)
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.disc, np.array([-1,0.5,1])))

        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        self.contnode.merge_interval(index = 2)
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.disc, np.array([-1,-0.5,1])))

        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        self.contnode.merge_interval(index = 3)
        self.assertIsNone(np.testing.assert_array_equal(self.contnode.disc, np.array([-1,-0.5,0.5])))

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
        error = self.contnode.compute_error(probs)
        new_disc, added_index, removed_indices = self.contnode.update_intervals(probs)
        self.assertIsNone(np.testing.assert_array_equal(new_disc, self.contnode.disc))
        self.assertIsNone(np.testing.assert_array_equal(new_disc, np.array([-1,-0.5,0.5,1,3,5,10])))
        self.assertEqual(added_index, 3, msg=f"{error}")
        self.assertIsNone(np.testing.assert_array_equal(removed_indices, np.array([])))

    def test_merge_interval_index_error(self):
        self.contnode.set_discretization(np.array([-1,-0.5,0.5,1]))
        self.assertRaises(IndexError, self.contnode.merge_interval(index = 4))


    def test_build_tabular_cpd(self):
        pass

if __name__ == '__main__':
    unittest.main()