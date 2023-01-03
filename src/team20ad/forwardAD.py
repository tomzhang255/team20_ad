import numpy as np

from team20ad.elementary import *



class ForwardAD:
    """Forward Mode Automatic Differentiation.

    Parameters
    ------
    var_dict: dict
        a dictionary of variables and their corresponding values
    func_list: str or list of str
        (a list of) function(s) encoded as string(s)

    Attributes
    ------
    func_evals: numpy.array
        the evaluation of function(s) at the given point 
    Dpf: numpy.array
        derivatives of function(s) evaluated at the given point

    Examples
    --------
    >>> v = {'x': 1, 'y': 2, 'z': 3}
    >>> f = 'tan(x) + exp(y) + sqrt(z)'
    >>> ad = ForwardAD(v, f)
    >>> ad()
    ===== Forward AD =====
    Vars: {'x': 1, 'y': 2, 'z': 3}
    Funcs: ['tan(x) + exp(y) + sqrt(z)']
    -----
    Func evals: [10.67851463115443]
    Gradient:
    [[3.42551882 7.3890561  0.28867513]]

    >>> var_dict = {'x': 1, 'y': 1}
    >>> func_list = ['x**2 + y**2', 'exp(x + y)']
    >>> ad = ForwardAD(var_dict, func_list)
    >>> ad()
    ===== Forward AD =====
    Vars: {'x': 1, 'y': 1}
    Funcs: ['x**2 + y**2', 'exp(x + y)']
    -----
    Func evals: [2, 7.38905609893065]
    Gradient:
    [[2.        2.       ]
     [7.3890561 7.3890561]]
    """

    def __init__(self, var_dict, func_list):
        # type checks
        if not isinstance(var_dict, dict):
            raise TypeError("var_dict should be a dictionary.")

        if isinstance(func_list, list):
            for f in func_list:
                if not isinstance(f, str):
                    raise TypeError("func_list should be a string or a list of strings.")
        elif not isinstance(func_list, str):
            raise TypeError("func_list should be a string or a list of strings.")

        # var inits
        elem_funcs = ['sqrt', 'exp', 'log', 'sin', 'cos', 'tan',
                      'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh']
        self.var_dict = var_dict

        if isinstance(func_list, list):
            self.func_list = func_list
        else:
            self.func_list = [func_list]

        vars = list(self.var_dict.keys())

        self.func_evals = []
        self.Dpf = np.zeros((len(self.func_list), len(self.var_dict)))
        i = 0  # a helper counter to determine which partial deriv to take

        for _ in self.var_dict:
            for var in self.var_dict:
                if var == vars[i]:
                    dual = 1
                else:
                    dual = 0
                exec(f"{var} = DualNumber({self.var_dict[var]}, {dual})")

            for j in range(0, len(self.func_list)):
                for f in elem_funcs:
                    func = self.func_list[j]
                    if f in self.func_list[j]:
                        break
                self.func_evals.append(eval(func).real)  # primal trace
                self.Dpf[j, i] = eval(func).dual  # tangent trace

            i += 1

        self.func_evals = self.func_evals[:len(self.func_list)]

    def __call__(self):
        out = "===== Forward AD =====\n"
        out += f"Vars: {self.var_dict}\n"
        out += f"Funcs: {self.func_list}\n"
        out += f"-----\n"
        out += f"Func evals: {self.func_evals}\n"
        out += f"Gradient:\n"
        out += f"{self.Dpf}"
        print(out)
