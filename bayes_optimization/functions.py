#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import Any, Dict, List, Optional

from ax.utils.measurement.synthetic_functions import (
    Hartmann6,
    Aug_Hartmann6,
    Branin,
    Aug_Branin,
    from_botorch,
    SyntheticFunction,
)

from botorch.test_functions import synthetic as botorch_synthetic


def get_function_id(function):
    return function.name + str(function.required_dimensionality)


class Sin(SyntheticFunction):

    _required_dimensionality = 1
    _domain = [(0, 2 * np.pi)]
    _minimums = [(1.5 * np.pi)]
    _maximums = [(0.5 * np.pi)]
    _fmin = -1
    _fmax = 1
    
    def _f(self, X:np.ndarray) -> float:
        return float(np.sin(X))


class SinWithOffsetMinus1p2(SyntheticFunction):

    _offset = -1.2
    
    _required_dimensionality = 1
    _domain = [(0, 2 * np.pi)]
    _minimums = [(1.5 * np.pi - _offset)]
    _maximums = [(0.5 * np.pi - _offset)]
    _fmin = -1
    _fmax = 1
    
    def _f(self, X:np.ndarray) -> float:
        return float(np.sin(X + self._offset))


class FlipedHartmann6(SyntheticFunction):
    """Fliped Hartmann6 function (6-dimensional with 1 global maximum).
    """

    _required_dimensionality = 6
    _domain = [(0, 1) for i in range(6)]
    _maximums = [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)]
    _fmin = 10.0 - 0.0
    _fmax = 10.0 - (-3.32237)
    _alpha = np.array([1.0, 1.2, 3.0, 3.2])
    _A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    _P = 10 ** (-4) * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )

    # @copy_doc(SyntheticFunction._f)
    def _f(self, X: np.ndarray) -> float:
        y = 0.0
        for j, alpha_j in enumerate(self._alpha):
            t = 0
            for k in range(6):
                t += self._A[j, k] * ((X[k] - self._P[j, k]) ** 2)
            y -= alpha_j * np.exp(-t)
        return 10.0 - float(y)


class AlignedHartmann6(SyntheticFunction):
    """Aligned Hartmann6 function (6-dimensional with 1 global minimum).

    Define a aligned version of synthetic function so that obj value is
    always positive. Therefore we can easily find a control arm with positive
    obj value. In this way, if we want to find minimizer of synthetic function
    we just need to minimize the ratio of experimental arm and control arm.

    Hartmann6 has max less than zero; need to do aligned.
    """

    _required_dimensionality = 6
    _domain = [(0, 1) for i in range(6)]
    _minimums = [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)]
    _fmin = 0.0 + 1
    _fmax = 3.32237 + 1
    _alpha = np.array([1.0, 1.2, 3.0, 3.2])
    _A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    _P = 10 ** (-4) * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )

    # @copy_doc(SyntheticFunction._f)
    def _f(self, X: np.ndarray) -> float:
        y = 0.0
        for j, alpha_j in enumerate(self._alpha):
            t = 0
            for k in range(6):
                t += self._A[j, k] * ((X[k] - self._P[j, k]) ** 2)
            y -= alpha_j * np.exp(-t) 
        return float(y + 3.32237 + 1)


class AlignedAugHartmann6(AlignedHartmann6):
    """Aligned Augmented Hartmann6 function
    
    (7-dimensional with 1 global minimum).
    """

    _required_dimensionality = 7
    _domain = [(0, 1) for i in range(7)]
    # pyre-fixme[15]: `_minimums` overrides attribute defined in `Hartmann6`
    #  inconsistently.
    _minimums = [
        (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573, 1.0)]
    _fmin = -3.32237 + 3.32237+1
    _fmax = 0 + 3.32237+1

    # @copy_doc(SyntheticFunction._f)
    def _f(self, X: np.ndarray) -> float:
        y = 0.0
        alpha_0 = self._alpha[0] - 0.1 * (1 - X[-1])
        for j, alpha_j in enumerate(self._alpha):
            t = 0
            for k in range(6):
                t += self._A[j, k] * ((X[k] - self._P[j, k]) ** 2)
            if j == 0:
                y -= alpha_0 * np.exp(-t)
            else:
                y -= alpha_j * np.exp(-t)
        return float(y+3.32237+1)


get_function_by_name = {
    # Custom aligned functions.
    "Sin": Sin(),
    "SinWithOffsetMinus1p2": SinWithOffsetMinus1p2(),
    "FlipedHartmann6": FlipedHartmann6(),
    "AlignedHartmann6": AlignedHartmann6(),
    "AlignedAugHartmann6": AlignedAugHartmann6(),
    # Ax support.
    "hartmann6": Hartmann6(),   # d: 6; [(0, 1) for i in range(6)]
    "AugHartmann6": Aug_Hartmann6(),   # d: 7; [(0, 1) for i in range(7)]
    "branin": Branin(),     # d: 2; [(-5, 10), (0, 15)]
    "AugBranin": Aug_Branin(),     # d: 3; [(-5, 10), (0, 15), (0, 1)]
    
    # From Botorch
    # # dim; _optimal_value; _bounds; _optimizers;
    # (2); 0.0; [(-32.768, 32.768),(-32.768, 32.768)]; [(0.0, 0.0)]
    "ackley2": from_botorch(botorch_synthetic_function=botorch_synthetic.Ackley(dim=2)),
    "ackley3": from_botorch(botorch_synthetic_function=botorch_synthetic.Ackley(dim=3)),
    "ackley4": from_botorch(botorch_synthetic_function=botorch_synthetic.Ackley(dim=4)),
    "ackley7": from_botorch(botorch_synthetic_function=botorch_synthetic.Ackley(dim=7)),
    # 2; 0.0; [(-4.5, 4.5), (-4.5, 4.5)]; [(3.0, 0.5)]
    "beale": from_botorch(botorch_synthetic_function=botorch_synthetic.Beale()),
    # 2; 0.0; [(-15.0, -5.0), (-3.0, 3.0)]; [(-10.0, 1.0)]
    "bukin": from_botorch(botorch_synthetic_function=botorch_synthetic.Bukin()),
    # 8; 0.8; [(-1.0, 1.0) for _ in range(8)]; [tuple(0.0 for _ in range(8))]
    "cosine8": from_botorch(botorch_synthetic_function=botorch_synthetic.Cosine8()),
    # 2; -1.0; [(-5.12, 5.12), (-5.12, 5.12)]; [(0.0, 0.0)]
    "dropwave": from_botorch(botorch_synthetic_function=botorch_synthetic.DropWave()),
    # (2); 0.0; [(-10.0, 10.0), (-10.0, 10.0)]; [tuple(math.pow(2.0, -(1.0 - 2.0 ** (-(i - 1)))) for i in range(2))]
    "dixonprice": from_botorch(botorch_synthetic_function=botorch_synthetic.DixonPrice()),
    # 2; -959.6407; [(-512.0, 512.0), (-512.0, 512.0)]; [(512.0, 404.2319)]
    "eggholder": from_botorch(botorch_synthetic_function=botorch_synthetic.EggHolder()),
    # (2); 0.0; [(-600.0, 600.0), (-600.0, 600.0)]; [(0.0, 0.0)]
    "griewank3": from_botorch(botorch_synthetic_function=botorch_synthetic.Griewank(dim=3)),
    # 2; -19.2085; [(-10.0, 10.0)]; [(8.05502, 9.66459),(-8.05502, -9.66459),(-8.05502, 9.66459),(8.05502, -9.66459),]
    # botorch has a bug for this function with 2 dim and 1 bounds.
    # holdertable = from_botorch(botorch_synthetic_function=botorch_synthetic.HolderTable())
    # (2); 0.0; [(-10.0, 10.0) for _ in range(self.dim)]; [tuple(1.0 for _ in range(self.dim))]
    "levy4": from_botorch(botorch_synthetic_function=botorch_synthetic.Levy(dim=4)),
    # 2; {2: -1.80130341, 5: -4.687658, 10: -9.66015}; [(0.0, math.pi) for _ in range(self.dim)]; {2: [(2.20290552, 1.57079633)]}
    "michalewicz": from_botorch(botorch_synthetic_function=botorch_synthetic.Michalewicz()),
    # (4); 0.0; [(-4.0, 5.0) for _ in range(self.dim)];
    "powell": from_botorch(botorch_synthetic_function=botorch_synthetic.Powell()),
    # (2); 0.0; [(-5.12, 5.12) for _ in range(self.dim)]; 
    "rastrigin5": from_botorch(botorch_synthetic_function=botorch_synthetic.Rastrigin(dim=5)),
    # (2); 0.0; [(-5.0, 10.0) for _ in range(self.dim)]
    "rosenbrock8": from_botorch(botorch_synthetic_function=botorch_synthetic.Rosenbrock(dim=8)),
    # 4; -10.536443; [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]; [(4.000747, 3.99951, 4.00075, 3.99951)]
    "shekel": from_botorch(botorch_synthetic_function=botorch_synthetic.Shekel()),
    # 2; -1.0316; [(-3.0, 3.0), (-2.0, 2.0)]; [(0.0898, -0.7126), (-0.0898, 0.7126)]
    "sixhumpcamel": from_botorch(botorch_synthetic_function=botorch_synthetic.SixHumpCamel()),
    # (2); -39.166166 * self.dim; [(-5.0, 5.0) for _ in range(self.dim)]; 
    "styblinskitang9": from_botorch(botorch_synthetic_function=botorch_synthetic.StyblinskiTang(dim=9)),
    # 2; 0.0; [(-5.0, 5.0), (-5.0, 5.0)]; [(0.0, 0.0)]
    "threehumpcamel": from_botorch(botorch_synthetic_function=botorch_synthetic.ThreeHumpCamel()),
}
