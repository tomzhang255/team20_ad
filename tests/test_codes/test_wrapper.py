import sys
sys.path.append("./src/")

from team20ad.wrapperAD import *
import numpy as np


class TestAD:

    def test_AD(self):
        vars = {'x': 0.5, 'y': 4}
        fcts = ['cos(x) + y ** 2', '2 * log(y) - sqrt(x)/3',
                'sqrt(x)/3', '3 * sinh(x) - 4 * arcsin(x) + 5']
        z = AD(vars, fcts)

        assert np.array_equal(np.around(z.func_evals, 4),
                              np.array([16.8776, 2.5369, 0.2357, 4.4689]))
        assert np.array_equal(np.around(z.Dpf, 4),
                              np.array([[-0.4794, 8.], [-0.2357, 0.5], [0.2357, 0.], [-1.2359, 0.]]))

    def test_repr_str(self):
        vars = {'x': 0.5, 'y': 4}
        fcts = ['cos(x) + y ** 2', '2 * log(y) - sqrt(x)/3',
                'sqrt(x)/3', '3 * sinh(x) - 4 * arcsin(x) + 5']
        z = AD(vars, fcts)
        assert isinstance(z.__str__(), str)
        assert isinstance(z.__repr__(), str)


def test_call(capfd):
    vars = {'x': 0.5, 'y': 4}
    fcts = ['cos(x) + y ** 2', '2 * log(y) - sqrt(x)/3',
            'sqrt(x)/3', '3 * sinh(x) - 4 * arcsin(x) + 5']
    z = AD(vars, fcts)
    z()  # outputs to std out
    out, err = capfd.readouterr()
    assert out is not None
