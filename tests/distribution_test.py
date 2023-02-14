import unittest
from hydipy import Distributions

class Test_NormalDistributionKnownParameters(unittest.TestCase):
    def setUp(self):
        self.dist = Distributions.NormalDistribution(5,1)
    def test_normal_cdf(self):
        self.assertEqual(self.dist.cdf(5.), 0.5)
        self.assertAlmostEqual(self.dist.cdf(7.), 0.9772499)
    def test_normal_ppf(self):
        self.assertEqual(self.dist.ppf(0.5), 5.)
        self.assertAlmostEqual(self.dist.ppf(0.95), 6.64485362)

class Test_NormalDistributionUnknownMean(unittest.TestCase):
    def setUp(self):
        self.dist = Distributions.NormalDistribution("a",1)
    def test_normal_cdf(self):
        self.assertEqual(self.dist.cdf(5.,mu=5.), 0.5)
        self.assertAlmostEqual(self.dist.cdf(7.,mu=5.), 0.9772499)
    def test_normal_ppf(self):
        self.assertEqual(self.dist.ppf(0.5,mu=5.), 5.)
        self.assertAlmostEqual(self.dist.ppf(0.95,mu=5.), 6.64485362)

class Test_NormalDistributionUnknownStdDev(unittest.TestCase):
    def setUp(self):
        self.dist = Distributions.NormalDistribution(5.,"b")
    def test_normal_cdf(self):
        self.assertEqual(self.dist.cdf(5.,sigma=1.), 0.5)
        self.assertAlmostEqual(self.dist.cdf(7.,sigma=1.), 0.9772499)
    def test_normal_ppf(self):
        self.assertEqual(self.dist.ppf(0.5,sigma=1.), 5.)
        self.assertAlmostEqual(self.dist.ppf(0.95,sigma=1.), 6.64485362)

class Test_NormalDistributionUnknownMeanStdDev(unittest.TestCase):
    def setUp(self):
        self.dist = Distributions.NormalDistribution("a","b")
    def test_normal_cdf(self):
        self.assertEqual(self.dist.cdf(5.,mu=5.,sigma=1.), 0.5)
        self.assertAlmostEqual(self.dist.cdf(7.,mu=5.,sigma=1.), 0.9772499)
    def test_normal_ppf(self):
        self.assertEqual(self.dist.ppf(0.5,mu=5.,sigma=1.), 5.)
        self.assertAlmostEqual(self.dist.ppf(0.95,mu=5.,sigma=1.), 6.64485362)

if __name__ == '__main__':
    unittest.main()