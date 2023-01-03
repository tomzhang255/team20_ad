from .forwardAD import ForwardAD
from .reverseAD import ReverseAD


class AD:
    """Automatic Differentiation wrapper that a mode can be specified. 

    If the mode is left unspecified by the user, it automatically determines 
    which mode to use based on the number of independent variables 
    and the number of functions to differentiate.

    Parameters
    ------
    var_dict: dict
        a dictionary of variables and their corresponding values
    func_list: str or list of str
        (a list of) function(s) encoded as string(s)
    mode: {None, "forward", "f", "reverse", "r"}
        string indicating mode of AD. Default is None.

    Attributes
    ------
    func_evals: numpy.array
        the evaluation of function(s) at the given point 
    Dpf: numpy.array
        derivatives of function(s) evaluated at the given point
    res: ForwardAD or ReverseAD objects
        ForwardAD or ReverseAD objects that the AD instance delegates diffirentiation tasks to

    Examples
    --------
    >>> var_dict = {'x': 1, 'y': 1}
    >>> func_list = ['x**2 + y**2', 'exp(x + y)']
    >>> ad = AD(var_dict, func_list)
    Number of variables <= number of functions: forward mode by default.
    >>> ad()
    ===== Forward AD =====
    Vars: {'x': 1, 'y': 1}
    Funcs: ['x**2 + y**2', 'exp(x + y)']
    -----
    Func evals: [2, 7.38905609893065]
    Gradient:
    [[2.        2.       ]
     [7.3890561 7.3890561]]

    >>> var_dict = {'x': 1, 'y': 2, 'z': 3}
    >>> func_list = ['tan(x) + exp(y) + sqrt(z)']
    >>> ad = AD(var_dict, func_list)
    Number of variables > number of functions: reverse mode by default.
    >>> ad()
    ===== Reverse AD =====
    Vars: {'x': 1, 'y': 2, 'z': 3}
    Funcs: ['tan(x) + exp(y) + sqrt(z)']
    -----
    Func evals: [10.67851463115443]
    Derivatives:
    [[3.42551882 7.3890561  0.28867513]]

    >>> v = {'x': 1, 'y': 2}
    >>> f = ['x**2 + y**2', 'exp(x + y)', 'tan(x + y) * sqrt(y)']
    >>> ad = AD(v, f, mode='r')
    >>> ad()
    ===== Reverse AD =====
    Vars: {'x': 1, 'y': 2}
    Funcs: ['x**2 + y**2', 'exp(x + y)', 'tan(x + y) * sqrt(y)']
    -----
    Func evals: [5.0, 20.085536923187668, -0.20159125448504428]
    Derivatives:
    [[ 2.          4.        ]
     [20.08553692 20.08553692]
     [ 1.4429497   1.39255189]]
    """
    def __init__(self, var_dict, func_list, mode = None):
        # check mode param valid
        if (mode is not None) and (mode not in ("forward", "f", "reverse", "r")):
            raise ValueError(f"Mode can be either forward, f, reverse, r, or None.") 
        
        self.mode = mode
        if self.mode is None: # if None, choose mode based on the criterion mentioned above
            num_var = len(var_dict)
            num_func = 1  # case: func_list is one string
            if isinstance(func_list, list):
                num_func = len(func_list)

            if num_var <= num_func:
                self.mode = "forward"
                print('Number of variables <= number of functions: forward mode by default.')
            else:
                self.mode = "reverse"
                print('Number of variables > number of functions: reverse mode by default.')

        if self.mode in ("forward", "f"):
            self.res = ForwardAD(var_dict, func_list)
        else:
            self.res = ReverseAD(var_dict, func_list)

        self.func_evals = self.res.func_evals
        self.Dpf = self.res.Dpf

    def __call__(self):
        return self.res.__call__()
