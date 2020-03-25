import unittest as ut
import numpy as np
from ..calculator import CofactorCalculator, Lattice

class TestCofactorCalculator(ut.TestCase):
    def setUp(self):
        tetr = Lattice(5.128, 5.128, 5.224, 90)
        mono = Lattice(5.203, 5.217, 5.388, 98.91)
        self.calc = CofactorCalculator(tetr, mono)

    def test_cofactors(self):
        cofactors = self.calc.get_cofactors()

        self.assertAlmostEqual(cofactors['A'], 1.017356, places=6)
        self.assertAlmostEqual(cofactors['B'], 0.998660, places=6)
        self.assertAlmostEqual(cofactors['C'], 1.017356, places=6)


if __name__ == '__main__':
    ut.main()
        
