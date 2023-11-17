import unittest
import numpy as np
from numpy.testing import assert_array_equal
from pyTensor.decomposition import fnorm
from pyTensor.tensorclass import astensor, unfold

m1 = np.loadtxt("../m1.txt", dtype=np.int32)
m2 = np.loadtxt("../m2.txt", dtype=np.int32)
tenseur = np.concatenate((m1[..., None], m2[..., None]), axis=2)  # [..., None] crée un nouvel axe vide


class TestHelperFunctions(unittest.TestCase):

    def test_fnorm_matrices(self):
        delta = 1e-5
        m1norm = round(fnorm(m1), 5)
        m2norm = round(fnorm(m2), 5)
        self.assertAlmostEqual(m1norm, 31.32092, delta=delta)
        self.assertAlmostEqual(m2norm, 31.14482, delta=delta)

    def test_fnorm_tenseur(self):
        delta = 1e-5
        tsr_norm = fnorm(tenseur)
        self.assertAlmostEqual(tsr_norm, 44.17013, delta=delta)

    def test_unfold_func(self):
        unfolded132 = np.loadtxt("../unfolded132.txt", dtype=np.float32)
        tnsr = astensor(tenseur)
        unfolded_matrice = unfold(tnsr, np.asarray([0, 2]), np.asarray([1]))
        assert_array_equal(unfolded132, unfolded_matrice)
        # diff = np.sum(unfolded132 != unfolded_matrice)
        # assert diff == 0, f"{diff} elements were different as a result of the unfold function"


if __name__ == '__main__':
    unittest.main()
