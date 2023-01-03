import sys
sys.path.append("./src/")

from team20ad.reverseAD import *
import numpy as np
import pytest


def test_node_init():
    x = Node(10)
    assert x.var == 10
    assert x.partial() == 1

    x = Node(10.0)
    assert x.var == 10
    assert x.partial() == 1

    with pytest.raises(TypeError):
        x = Node("some string")


def test_node_get_derivatives():
    x = Node(3)
    z = Node(5)
    y = x * z
    assert y.g_derivatives([x, z])[0] == 15
    assert y.g_derivatives([x, z])[1][0] == 5
    assert y.g_derivatives([x, z])[1][1] == 3


def test_node_partial():
    pass


def test_node_str():
    x = Node(5)
    assert str(x) == "value = 5\nderivative = 1"


def test_node_repr():
    x = Node(5)
    print(repr(x))
    assert repr(x) == "Node(5)"


def test_node_add():
    # add two Node objects
    x = Node(3)
    y = x + Node(5)
    assert y.var == 8
    assert y.partial() == 1

    # radd float
    x = Node(1)
    y = 2.0 + x
    assert y.var == 3
    assert y.partial() == 1.0

    # add float
    x = Node(1)
    y = x + 2.
    assert y.var == 3
    assert y.partial() == 1.0

    # radd float, then Node
    x = Node(1)
    y = 2.0 + x + Node(2)
    assert y.var == 5
    assert y.partial() == 1

    # add invalid type
    with pytest.raises(TypeError):
        x = Node(1)
        y = x + "string"

    with pytest.raises(TypeError):
        x = Node(1)
        y = "string" + x


def test_node_neg():
    x = Node(5)
    y = -x
    assert y.var == -5
    assert y.partial() == 1
    assert x.partial() == -1


def test_node_sub():
    # sub nodes
    x = Node(1)
    y = x - Node(2)
    assert y.var == -1
    assert y.partial() == 1

    # sub float
    x = Node(1)
    y = x - 2.0
    assert y.var == -1
    assert y.partial() == 1

    # rsub float
    x = Node(1)
    y = 2.0 - x
    assert y.var == 1
    assert y.partial() == 1
    assert x.partial() == -1

    # sub invalid type
    with pytest.raises(TypeError):
        x = Node(999)
        y = x - "invalid"


def test_node_mul():
    # mul
    x = Node(3)
    y = x * Node(5)
    assert y.var == 15
    assert y.partial() == 1
    assert x.partial() == 5

    # mul float
    x = Node(3)
    y = x * 5.0
    assert y.var == 15
    assert y.partial() == 1
    assert x.partial() == 5

    # rmul float
    x = Node(3)
    y = 5.0 * x
    assert y.var == 15
    assert x.partial() == 5
    assert y.partial() == 1

    x = Node(3)
    y = x * -5.0
    assert y.var == -15
    assert x.partial() == -5
    assert y.partial() == 1

    x = Node(3)
    y = -5.0 * x
    assert y.var == -15
    assert x.partial() == -5
    assert y.partial() == 1

    # mul invalid
    with pytest.raises(TypeError):
        x = Node(1)
        y = x * "invalid"


def test_node_truediv():
    # truediv nodes
    x = Node(1)
    y = Node(2) / x
    assert y.var == 2
    assert y.partial() == 1
    assert x.partial() == -2

    # rtruediv float
    x = Node(1)
    y = 5. / x
    assert y.var == 5
    assert x.partial() == -5

    # truediv float
    x = Node(1)
    y = x / 5.
    assert y.var == 1/5
    assert x.partial() == 1/5
    assert y.partial() == 1

    # truediv invalid
    with pytest.raises(TypeError):
        x = Node(1)
        y = x / "invalid"

    with pytest.raises(TypeError):
        x = Node(1)
        y = "string" / x


def test_node_lt():
    x = Node(10)
    y = Node(11)
    assert (x < y) == True

    x = Node(99)
    assert (x < 99) == False

    with pytest.raises(TypeError):
        x = Node(1)
        _ = "string" < x


def test_node_gt():
    x = Node(1)
    y = Node(99)
    assert (y > x) == True

    x = Node(1)
    assert (x > 99) == False

    with pytest.raises(TypeError):
        x = Node(1)
        _ = "string" > x


def test_node_le():
    x = Node(0)
    y = Node(1)
    assert (x <= y) == True

    x = Node(2)
    assert (x <= 2) == True

    with pytest.raises(TypeError):
        x = Node(2)
        _ = "string" <= x


def test_node_ge():
    x = Node(3)
    y = Node(4)
    assert (y >= x) == True

    x = Node(10)
    assert (x >= 10) == True

    with pytest.raises(TypeError):
        x = Node(0)
        _ = "string" >= x


def test_node_eq():
    x = Node(0)
    y = Node(1)
    assert (y == x) == False

    with pytest.raises(TypeError):
        x = Node(0)
        _ = x == 0


def test_node_ne():
    x = Node(99)
    y = Node(100)
    assert (y != x) == True

    with pytest.raises(TypeError):
        x = Node(100)
        _ = x != 'wrong type'


def test_node_abs():
    x = abs(Node(99))
    assert x.var == 99
    assert x.partial() == 1

    y = abs(Node(-88))
    assert y.var == 88
    assert y.partial() == 1


def test_node_pow():
    x = Node(2)
    y = x ** 3
    assert y.var == 8
    assert x.partial() == 12
    assert y.partial() == 1

    x = Node(2)
    y = x ** -3
    assert y.var == 2 ** -3
    assert x.partial() == -3 * (2 ** -4)
    assert y.partial() == 1

    x = Node(2)
    y = x ** x
    assert y.var == 2 ** 2
    assert x.partial() == 2 ** 2 * (np.log(2) * 1 / 1 + 2 / 2)
    assert y.partial() == 1

    x = Node(2)
    y = x ** (x * 2)
    assert y.var == 2 ** (2 * 2)
    assert x.partial() == 2 ** (2 * 2) * (np.log(2) * 2 / 1 + (2 * 2) / 2)
    assert y.partial() == 1


def test_node_rpow():
    x = Node(3)
    y = 2 ** x
    assert y.var == 2 ** 3
    assert x.partial() == 2 ** 3 * np.log(2)
    assert y.partial() == 1

    with pytest.raises(Exception):
        x = Node(2)
        y = -4 ** x
        assert y.var == 4 ** 2

    x = Node(2)
    y = 4 ** (x * 2)
    assert y.var == 4 ** (2 * 2)
    assert x.partial() == 2 ** (4 * 2 + 1) * np.log(4)

    with pytest.raises(ValueError):
        x = Node(2)
        y = "string" ** x


def test_node_log():
    x = Node(2)
    y = Node.log(x)
    assert y.var == np.log(2)
    assert x.partial() == 1 / 2
    assert y.partial() == 1

    x = Node(2)
    y = Node.log(2 * x)
    assert y.var == 2 * np.log(2)
    assert x.partial() == 1 / 2
    assert y.partial() == 1

    with pytest.raises(TypeError):
        x = Node(-1)
        y = Node.log(x)

    with pytest.raises(TypeError):
        x = Node(-1)
        y = Node.log(3 * x)

    with pytest.raises(TypeError):
        y = Node.log(1)


def test_node_sqrt():
    x = Node(2)
    y = Node.sqrt(x)
    assert y.var == np.sqrt(2)
    assert x.partial() == 1/2 * 2 ** (-1/2)

    x = Node(2)
    y = Node.sqrt(2 * x)
    assert y.var == np.sqrt(2 * 2)
    assert x.partial() == 1 / np.sqrt(2) * 2 ** (-1/2)
    with pytest.raises(ValueError):
        x = Node(-2)
        y = Node.sqrt(x)

    with pytest.raises(TypeError):
        y = Node.sqrt("string")

    with pytest.raises(TypeError):
        y = Node.sqrt(2)


def test_node_exp():
    x = Node(2)
    y = Node.exp(x)
    assert y.var == np.exp(2)
    assert x.partial() == np.exp(2)

    x = Node(2)
    y = Node.exp(2 * x)
    assert y.var == np.exp(2 * 2)
    assert x.partial() == 2 * np.exp(2 * 2)

    with pytest.raises(TypeError):
        y = Node.exp("string")

    y = Node.exp(2)
    assert y == np.exp(2)


def test_node_sin():
    x = Node(np.pi/4)
    y = Node.sin(x)
    assert y.var == np.sin(np.pi/4)
    assert x.partial() == np.cos(np.pi/4)

    x = Node(np.pi/4)
    y = Node.sin(3 * x)
    assert y.var == np.sin(3 * np.pi/4)
    assert x.partial() == 3 * np.cos(3 * np.pi/4)

    with pytest.raises(TypeError):
        y = Node.sin("string")

    y = Node.sin(np.pi/2)
    assert y == np.sin(np.pi/2)


def test_node_cos():
    x = Node(np.pi/4)
    y = Node.cos(x)
    assert y.var == np.cos(np.pi/4)
    assert x.partial() == -np.sin(np.pi/4)

    with pytest.raises(TypeError):
        y = Node.cos("string")

    y = Node.cos(np.pi)
    assert y == np.cos(np.pi)


def test_node_tan():
    x = Node(np.pi/3)
    y = Node.tan(x)
    assert y.var == np.tan(np.pi/3)
    assert x.partial() == 1/np.cos(np.pi/3)**2

    with pytest.raises(TypeError):
        y = Node.tan("string")

    y = Node.tan(np.pi/3)
    assert y == np.tan(np.pi/3)


def test_node_arcsin():
    x = Node(1/4)
    y = Node.arcsin(x)
    assert y.var == np.arcsin(1/4)
    assert x.partial() == 1 / np.sqrt(1 - (1/4) ** 2)

    with pytest.raises(TypeError):
        y = Node.arcsin("string")

    with pytest.raises(TypeError):
        x = Node(-4)
        y = Node.arcsin(x)

    y = Node.arcsin(1/4)
    assert y == np.arcsin(1/4)


def test_node_arccos():
    x = Node(1/4)
    y = Node.arccos(x)
    assert y.var == np.arccos(1/4)
    assert x.partial() == -1 / np.sqrt(1 - (1/4) ** 2)

    with pytest.raises(TypeError):
        y = Node.arccos("string")

    with pytest.raises(TypeError):
        x = Node(-4)
        y = Node.arccos(x)

    assert Node.arccos(1/4) == np.arccos(1/4)


def test_node_arctan():
    x = Node(1/4)
    y = Node.arctan(x)
    assert y.var == np.arctan(1/4)
    assert x.partial() == 1 / (1 + np.power(1/4, 2))

    with pytest.raises(TypeError):
        y = Node.arctan("string")

    assert Node.arctan(1/4) == np.arctan(1/4)


def test_node_sinh():
    x = Node(1/2)
    y = Node.sinh(x)
    assert y.var == np.sinh(1/2)
    assert x.partial() == np.cosh(1/2)

    with pytest.raises(TypeError):
        y = Node.sinh("string")

    assert Node.sinh(1/2) == np.sinh(1/2)


def test_node_cosh():
    x = Node(1/2)
    y = Node.cosh(x)
    assert y.var == np.cosh(1/2)
    assert x.partial() == np.sinh(1/2)

    with pytest.raises(TypeError):
        y = Node.cosh("string")

    assert Node.cosh(1/2) == np.cosh(1/2)


def test_node_tanh():
    x = Node(1/2)
    y = Node.tanh(x)
    assert y.var == np.tanh(1/2)
    assert y.partial() == 1
    with pytest.raises(TypeError):
        y = Node.tanh("string")

    assert Node.tanh(1/2) == np.tanh(1/2)


def test_node_logistic():
    x = Node(3)
    y = Node.logistic(x)
    assert y.var == 1 / (1 + np.exp(-3))
    assert x.partial() == 1 / (1 + np.exp(-3)) * (1 - 1 / (1 + np.exp(-3)))

    with pytest.raises(TypeError):
        y = Node.logistic("string")
