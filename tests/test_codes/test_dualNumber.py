import sys
sys.path.append("./src/")

import pytest
import numpy as np
from team20ad.dualNumber import DualNumber
from team20ad.elementary import *
from team20ad.forwardAD import *


class TestDualNumber:

    def test_initializer(self):
        with pytest.raises(TypeError):
            x = DualNumber("hello")

    def test_add_radd(self):
        x = DualNumber(3, 1)
        y = x + 3

        assert y.real == 6
        assert y.dual == 1

        x = DualNumber(3, 1)
        y = x + DualNumber(3, 1)

        assert y.real == 6
        assert y.dual == 2


    def test_mul_rmul(self):
        x = DualNumber(3, 1)
        y = x * 3

        assert y.real == 9
        assert y.dual == 3

        x = DualNumber(3)
        y = x * DualNumber(3)

        assert y.real == 9
        assert y.dual == 6

        x = DualNumber(3, 1)
        y = x * DualNumber(3, 0)

        assert y.real == 9
        assert y.dual == 3

        x = DualNumber(3, 1)
        y = DualNumber(3, 0) * x

        assert y.real == 9
        assert y.dual == 3

    def test_sub_rsub(self):
        x = DualNumber(3, 1)
        y = x - 3

        assert y.real == 0
        assert y.dual == 1

        x = DualNumber(3, 1)
        y = x - DualNumber(3, 1)

        assert y.real == 0
        assert y.dual == 0

        x = DualNumber(4, 1)
        y = x - DualNumber(3, 0)

        assert y.real == 1
        assert y.dual == 1

        x = DualNumber(3, 1)
        y = 3 - x

        assert y.real == 0
        assert y.dual == -1

        x = DualNumber(3, 1)
        y = DualNumber(3, 1) - x

        assert y.real == 0
        assert y.dual == 0

        x = DualNumber(3, 1)
        y = DualNumber(4, 0) - x

        assert y.real == 1
        assert y.dual == -1

    def test_truediv_rtruediv(self):
        x = DualNumber(3, 1)
        y = x / 3

        assert y.real == 1
        assert y.dual == 1 / 3

        x = DualNumber(3, 1)
        y = x / DualNumber(4, 1)

        assert y.real == 3 / 4
        assert y.dual == 1 / 16

        x = DualNumber(3, 1)
        y = x / DualNumber(4, 0)

        assert y.real == 3 / 4
        assert np.array_equal(y.dual, 1 / 4)

        x = DualNumber(3, 1)
        y = 2 / x

        assert y.real == 2 / 3
        assert y.dual == -2 / 9

        x = DualNumber(4, 1)
        y = DualNumber(3, 1) / x

        assert y.real == 3 / 4
        assert y.dual == 1 / 16

        x = DualNumber(4, 0)
        y = DualNumber(3, 1) / x

        assert y.real == 3 / 4
        assert np.array_equal(y.dual, 1 / 4)

    def test_neg(self):
        x = DualNumber(3, 1)
        y = -x
        assert y.real == -3
        assert y.dual == -1


    def test_eq(self):
        X = DualNumber(3, 1)
        Y = DualNumber(3, 1)
        flag = (X == Y)
        assert flag == True

        flag = (DualNumber(3, 1) == 3)
        assert flag == False

    def test_ne(self):
        X = DualNumber(3, 1)
        Y = DualNumber(3, 1)
        flag = (X != Y)
        assert flag == False

        flag = (DualNumber(3, 1) != 3)
        assert flag == True



    def test_lt(self):
        X = DualNumber(3, 1)
        Y = DualNumber(4, 1)
        flag = (X < Y)
        assert flag == True

        flag = (DualNumber(3, 1) < 3)
        assert flag == False
        assert X < 10
        
        
    def test_le(self):
        X = DualNumber(3, 1)
        Y = DualNumber(4, 1)
        flag = (X < Y)
        assert flag == True

        flag = (DualNumber(3, 1) < 3)
        assert flag == False
        assert X < 10

    def test_gt(self):
        X = DualNumber(3, 1)
        Y = DualNumber(4, 1)
        flag = (X > Y)
        assert flag == False

        flag = (DualNumber(3, 1) > 3)
        assert flag == False
        assert X > 1

    def test_ge(self):
        X = DualNumber(3, 1)
        Y = DualNumber(4, 1)
        flag = (X >= Y)
        assert flag == False

        flag = (DualNumber(3, 1) >= 3)
        assert flag == True
        assert X > 1

    def test_abs(self):
        y = abs(DualNumber(-3, -1))
        assert y.real == 3
        assert y.dual == 1

