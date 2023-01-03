import sys
sys.path.append("./src/")

import pytest
import numpy as np
from team20ad.dualNumber import DualNumber
from team20ad.elementary import *
from team20ad.forwardAD import *


def test_pow():
    # test DualNumber raise to a constant
    x = DualNumber(5)
    y = x ** 3

    assert y.real == 5 ** 3
    assert y.dual == 3 * (5 ** 2)
    with pytest.raises(TypeError):
        x**"2"

def test_pow_var():
    # test DualNumber raise to a DualNumber
    x = DualNumber(2)
    y = x ** (3 * x)

    assert y.real == 2 ** 6
    assert y.dual == 192 + 192 * np.log(2)

def test_pow_imaginary():
    # test pow when base < 0 and exponent < 1
    with pytest.raises(TypeError):
        x = DualNumber(-1)
        y = x ** -0.5

def test_rpow():
    # test constant raise to a DualNumber
    x = DualNumber(3)
    y = 5 ** x
    assert y.real == 5 ** 3
    assert y.dual == 125 * np.log(5)

def test_rpow_imaginary():
    # test constant raise to a DualNumber
    with pytest.raises(ValueError):
        x = DualNumber(0.5)
        y = (-2) ** x

def test_sqrt():
    # test square root of a DualNumber
    x = DualNumber(10.1)
    y = sqrt(x)
    assert y.real == np.sqrt(10.1)
    assert y.dual == 1 / (2 * (np.sqrt(10.1)))
    with pytest.raises(ValueError):
        sqrt(-1)
    with pytest.raises(TypeError):
        sqrt("-1")
        

def test_sqrt_constant():
    # test square root of a DualNumber
    x = 12
    y = sqrt(x)
    assert y == np.sqrt(12)

def test_sqrt_non_positive():
    # test square root of a non-positve DualNumber
    with pytest.raises(ValueError):
        x = DualNumber(-10.1)
        y = sqrt(x)

def test_exp():
    x = DualNumber(32)
    y = exp(x)
    assert y.real == np.exp(32)
    assert y.dual == np.exp(32)
    with pytest.raises(TypeError):
        exp("3")


def test_exp_constant():
    x = 32
    y = exp(x)
    assert y == np.exp(32)

def test_log(): 
    x = DualNumber(14)
    y = log(x)
    assert y.real == np.log(14)
    assert y.dual == 1 / 14

    x = DualNumber(8)
    y = log(x,2)
    assert y.real == np.log2(8)
    with pytest.raises(TypeError):
        log("2")


def test_log_constant():
    x = 14
    y = log(x)
    assert y == np.log(14)
    x = 8
    y = log(x,2)
    assert y == np.log2(8)

def test_log_non_positive():
    with pytest.raises(ValueError):
        x = DualNumber(-14)
        log(x)
    with pytest.raises(ValueError):
        log(-10)

def test_tangent_function():
    x = DualNumber(np.pi)
    f = tan(x)

    # However, if you specify a message with the assertion like this:
    # assert a % 2 == 0, "value was odd, should be even"
    # then no assertion introspection takes places at all and the message will be simply shown in the traceback.
    assert f.dual == 1

    x = DualNumber(3 * np.pi / 2)
    with pytest.raises(ValueError):
        f = tan(x)

    x = DualNumber(2)
    f = 3 * tan(x)

    assert f.real == 3 * np.tan(2) and np.round(f.dual, 4) == 17.3232

    x = DualNumber(np.pi)
    f = tan(x) * tan(x)

    assert np.round(f.dual, 5) == 0

    # checking a constant
    assert tan(3) == np.tan(3)
    with pytest.raises(TypeError):
        tan("2")

# can use these function below to run the code manually rather than with pytest
def test_arctangent_function():
    x = DualNumber(2)
    f = arctan(x)

    assert f.dual == .2 and np.round(f.real, 4) == 1.1071

    # check a constant
    assert arctan(3) == np.arctan(3)
    with pytest.raises(TypeError):
        arctan("2")

def test_sinh_function():
    x = DualNumber(2)
    f = 2 * sinh(x)

    assert np.round(f.dual, 4) == 7.5244 and np.round(f.real, 4) == 7.2537

    # check a constant
    assert sinh(3) == np.sinh(3)
    with pytest.raises(TypeError):
        sinh("2")

def test_cosh_function():
    x = DualNumber(4)
    f = 3 * cosh(x)
    assert np.round(f.real, 4) == 81.9247 and np.round(f.dual, 4) == 81.8698

    # check a constant
    assert cosh(3) == np.cosh(3)
    with pytest.raises(TypeError):
        cosh("2")

def test_tanh_function():
    x = DualNumber(3)
    f = 2 * tanh(x)
    assert np.round(f.real, 4) == 1.9901 and np.round(f.dual, 4) == 0.0197

    # checking a constant
    assert tanh(3) == np.tanh(3)
    with pytest.raises(TypeError):
        tanh("2")

def test_sin():
    x = DualNumber(0)
    f = sin(x)

    assert f.real == 0.0
    assert f.dual == 1.

    # check constant
    assert sin(2) == np.sin(2)
    with pytest.raises(TypeError):
        sin("2")

def test_cos():
    x = DualNumber(0)
    f = cos(x)
    assert f.real == 1.0
    assert f.dual == 0.

    # check constant
    assert cos(2) == np.cos(2)
    with pytest.raises(TypeError):
        cos("2")

def test_arcsin():
    x = DualNumber(0)
    f = arcsin(x)
    assert f.real == 0.0
    assert f.dual == 1.
    # -1<= x <=1

    with pytest.raises(ValueError):
        x = DualNumber(-2)
        f = arcsin(x)

    assert arcsin(0.5) == np.arcsin(0.5)
    with pytest.raises(TypeError):
        arcsin("2")

def test_arccos():
    x = DualNumber(0)
    f = arccos(x)
    assert np.round(f.real, 4) == 1.5708
    assert f.dual == -1.

    with pytest.raises(ValueError):
        x = DualNumber(2)
        f = arccos(x)

    assert arccos(0.5) == np.arccos(0.5)
    with pytest.raises(TypeError):
        arccos("2")

def test_logistic():
    x = DualNumber(100)
    y = logistic(x)
    assert y == DualNumber(1.0, 0.0)

    x = 100
    y = logistic(x)
    assert y == DualNumber(1.0, 0.0)
    
    with pytest.raises(TypeError):
        logistic('test')