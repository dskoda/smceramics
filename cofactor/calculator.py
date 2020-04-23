import numpy as np
from collections import namedtuple
from scipy import stats


RADIANS = np.pi / 180
Lattice = namedtuple("Lattice", "a b c beta")


class CofactorCalculator:
    def __init__(self, tetragonal, monoclinic):
        self.tetr = tetragonal
        self.mono = monoclinic
        #self._check_inputs()

    def _check_inputs(self):
        assert isinstance(self.tetr, Lattice)
        assert isinstance(self.mono, Lattice)
        np.testing.assert_almost_equal(self.tetr.a, self.tetr.b)
        np.testing.assert_almost_equal(self.tetr.beta, 90)
        assert not np.isclose(self.mono.beta, 90)

    def get_deformation_matrix_A(self):
        f11 = self.mono.b / self.tetr.a
        f22 = self.mono.c / self.tetr.a
        f23 = self.mono.a * np.cos(self.mono.beta * RADIANS) / self.tetr.c
        f33 = self.mono.a * np.sin(self.mono.beta * RADIANS) / self.tetr.c

        F = np.array([
            [f11, 0.0, 0.0],
            [0.0, f22, f23],
            [0.0, 0.0, f33]
        ])

        return F

    def get_deformation_matrix_B(self):
        f11 = self.mono.c / self.tetr.a
        f12 = self.mono.a * np.cos(self.mono.beta * RADIANS) / self.tetr.a
        f22 = self.mono.a * np.sin(self.mono.beta * RADIANS) / self.tetr.a
        f33 = self.mono.b / self.tetr.c

        F = np.array([
            [f11, f12, 0.0],
            [0.0, f22, 0.0],
            [0.0, 0.0, f33]
        ])

        return F

    def get_deformation_matrix_C(self):
        f11 = self.mono.a * np.sin(self.mono.beta * RADIANS) / self.tetr.a
        f22 = self.mono.b / self.tetr.a
        f31 = self.mono.a * np.cos(self.mono.beta * RADIANS) / self.tetr.a
        f33 = self.mono.c / self.tetr.c

        F = np.array([
            [f11, 0.0, 0.0],
            [0.0, f22, 0.0],
            [f31, 0.0, f33]
        ])

        return F

    @property
    def deformations(self):
        return {
            'A': self.get_deformation_matrix_A(),
            'B': self.get_deformation_matrix_B(),
            'C': self.get_deformation_matrix_C(),
        }

    def get_stretch_tensor(self, F):
        w, V = np.linalg.eig(F.T @ F)
        U = V @ (np.sqrt(w) * np.eye(3)) @ V.T

        return U

    def get_cofactor_correspondence(self, F):
        U = self.get_stretch_tensor(F)
        w, _ = np.linalg.eig(U)
        return sorted(w.tolist())[1]

    def get_volume_change_correspondence(self, F):
        U = self.get_stretch_tensor(F)
        return np.linalg.det(U)

    def get_cofactors(self):
        return {
            key + '_lambda': self.get_cofactor_correspondence(F)
            for key, F in self.deformations.items()
        }

    def get_volume_change(self):
        return {
            key + '_dV': self.get_volume_change_correspondence(F)
            for key, F in self.deformations.items()
        }
