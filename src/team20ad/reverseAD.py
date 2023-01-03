import numpy as np
import re

from .elementary import *

class ReverseAD:
    """Reverse Mode Automatic Differentiation.

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
    >>> ad = ReverseAD(v, f)
    >>> ad()
    ===== Reverse AD =====
    Vars: {'x': 1, 'y': 2, 'z': 3}
    Funcs: ['tan(x) + exp(y) + sqrt(z)']
    -----
    Func evals: [10.67851463115443]
    Derivatives:
    [[3.42551882 7.3890561  0.28867513]]

    >>> var_dict = {'x': 1, 'y': 2}
    >>> func_list = ['x**2 + y**2', 'exp(x + y)']
    >>> ad = ReverseMode(var_dict, func_list)
    >>> ad()
    ===== Reverse AD =====
    Vars: {'x': 1, 'y': 2}
    Funcs: ['x**2 + y**2', 'exp(x + y)']
    -----
    Func evals: [5.0, 20.085536923187668]
    Derivatives:
    [[ 2.          4.        ]
     [20.08553692 20.08553692]]      
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

        if isinstance(func_list, list):
            self.func_list = func_list
        else: # if a single string, convert it to list
            self.func_list = [func_list]
 
        self.func_evals = []
        self.Dpf = []
        self.var_dict = var_dict

        elem_funcs = ['sqrt', 'exp', 'log', 'sin', 'cos', 'tan',
                      'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh']

        for func in self.func_list:
            for i in elem_funcs:
                if i in func:
                    func = re.sub(i + r'\(', 'Node.' + i + '(', func)
                    func = re.sub('arcNode.', 'arc', func)

            for var_name, var_value in var_dict.items():
                exec(f'{var_name} = Node(float(var_value))')
            vals = eval(func)

            value_keys = str(list(var_dict.keys())).replace('\'','')
            v, d = eval(f'vals.g_derivatives({value_keys})')

            self.func_evals.append(v)
            self.Dpf.append(d)
        self.Dpf = np.array(self.Dpf)

    def __call__(self):
        out = "===== Reverse AD =====\n"
        out += f"Vars: {self.var_dict}\n"
        out += f"Funcs: {self.func_list}\n"
        out += f"-----\n"
        out += f"Func evals: {self.func_evals}\n"
        out += f"Derivatives:\n{self.Dpf}\n"
        print(out)


class Node():
    """Node object for supporting operations of reverse mode AD

    Attributes
    ------
    var : int or float
        a primal trace value to be stored at each node.
    child : list
        a list of all depending Nodes and their associated derivatives
    derivative : float 
        Representing the current evaluated derivative
    """
    def __init__(self, var):
        """ Node constructor.

        Parameter
        ------
        var : int or float
            a primal trace value to be stored at each node.

        Raises
        ------
        TypeError
            if an argument value is of unsupported type. 
        """
        if isinstance(var, int) or isinstance(var, float):
            self.derivative = None
            self.var = var
            self.child = []
            
        else:
            raise TypeError("Input must be int or float.")


    def __str__(self):
        """Returns a string representation of the Node instance.

        Returns
        ------
        str
            a string representation of the Node instance.
        """
        return f"value = {self.var}\nderivative = {self.partial()}"


    def __repr__(self):
        """Returns a representation of the Node instance.

        Returns
        ------
        str
            a representation of the Node instance.
        """
        return f"Node({self.var})"
        

    def g_derivatives(self, inputs):
        """
        Get derivatives for each variable in the function.

        Parameter
        ------
        inputs : list
            list of functions

        Returns
        ------
        var_val : int or float
            a variable which stores the function values.
        der_list : list
            a list of derivatives for each variable.
        """
        # self.der = 1
        v_val = self.var
        der_list = np.array([v_i.partial() for v_i in inputs])
        
        return v_val, der_list
            

    def partial(self):
        """Computes derivative for a variable used in the function."""
        if len(self.child) == 0:
            return 1
        if self.derivative is not None:
            return self.derivative
        else:
            self.derivative = sum([child.partial() * partial for child, partial in self.child])
            return self.derivative


    def __add__(self, other):
        """Returns a new Node instance as a result of the addition.

        Parameter
        ------
        other : Node, int, or float
            the instance to compute the sum with.

        Returns
        ------
        Node
            a new Node instance as a sum of the two instances.
        """
        try:
            new_add = Node(self.var + other.var)
            self.child.append((new_add, 1))
            other.child.append((new_add, 1))
           
            return new_add
        except: 
            if isinstance(other, int) or isinstance(other, float):
                new_add = Node(self.var + other)
                self.child.append((new_add, 1))
                return new_add
            else:
                raise TypeError("Not real number")


    def __radd__(self, other):
        """Returns a new Node instance as a result of the addition.

        As the add operation is commutative, this method delegates the operation
        to __add__().

        Parameter
        ------
        other : int, or float
            a scalar object to compute the sum with.

        Returns
        ------
        Node
            the sum of the two instances.
        """
        return self.__add__(other)


    def __sub__(self, other):
        """Returns a new Node instance as a result of the subtraction.

        Parameter
        ------
        other : Node, int, or float
            the instance to compute the difference with.

        Returns
        ------
        Node
            a new Node instance as a difference between the two instances.
        """
        return self.__add__(-other)


    def __rsub__(self, other):
        """Returns a new Node instance as a result of the (reverse) subtraction.

        Parameter
        ------
        other : int, or float
            the instance to compute the difference with.

        Returns
        ------
        Node
            a new Node instance as a difference between the two instances.
        """
        return (-self).__add__(other)


    def __mul__(self, other):
        """Returns a new Node instance as a result of the multiplication.

        Parameter
        ------
        other : Node, int, or float
            the instance to compute the product with.

        Returns
        ------
        Node
            a new Node instance as a product of the two instances.
        """
        try:
            new_mul = Node(other.var * self.var)
            self.child.append((new_mul, other.var))
            other.child.append((new_mul, self.var))
            return new_mul
        except:
            if isinstance(other, int) or isinstance(other, float):
                # other is not a Node and the multiplication could 
                # be completed if it is a real number
                new_mul = Node(other * self.var)
                self.child.append((new_mul, other))
                return new_mul
            else:
                raise TypeError("Input is not a real number.")
        
    def __rmul__(self, other):
        """Returns a new Node instance as a result of the multiplication.

        As the multiplication operation is commutative, this method delegates the operation
        to __mul__().

        Parameter
        ------
        other : int, or float
            a scalar object to compute the product with.

        Returns
        ------
        Node
            the product of the two instances.
        """
        return self.__mul__(other)


    def __truediv__(self, other):
        """Returns a new Node instance as a result of the division.

        Parameter
        ------
        other : Node, int, or float
            the instance to compute the division with.

        Returns
        ------
        Node
            a new Node instance as a division of the two instances.
        """
        try:
            new_div = Node(self.var / other.var)
            self.child.append((new_div,((1 * other.var - 0 * self.var) / other.var**2)))
            other.child.append((new_div, (-self.var/(other.var**2))))
            return new_div
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):
                new_div = Node(self.var / other)
                self.child.append((new_div,((1 * other - 0 * self.var) / other**2)))
                return new_div
            else:
                raise TypeError(f"{other} is invalid.")


    def __rtruediv__(self, other):
        """Returns a new Node instance as a result of the (reverse) division.

        Parameter
        ------
        other : int, or float
            the instance to compute the division with.

        Returns
        ------
        Node
            a new Node instance as a (reverse) division of the two instances.
        """
        try:
            new_div = Node(other.var / self.var)
            self.child.append((new_div, ((0 * self.var - other.var * 1) / self.var**2)))
            other.child.append((new_div, 1/self.var))
            return new_div
        except:
            if isinstance(other, int) or isinstance(other, float):
                new_div = Node(other / self.var)
                self.child.append((new_div, ((0 * self.var - other * 1) / self.var**2)))
                return new_div
            else:
                raise TypeError(f"Input {other} is not valid.")


    def __neg__(self):
        """Returns a new node instance as the negation of the Node instance.

        Returns
        ------
        Node
            a Node instance that has a negated value
        """
        new_neg = Node(-self.var)
        self.child.append((new_neg, -1))
        return new_neg

    
    def __lt__(self, other):
        """Compares two objects if the instance has value less than the other given instance of supported type.

        Parameter
        ------
        other : Node, int, or float
            The object to compare with.

        Returns
        ------
        bool
            True if less than the given object; and False, otherwise.
        """
        try:
            return self.var < other.var
        except AttributeError:
            return self.var < other


    def __gt__(self, other):
        """Compares two objects if the instance has value greater than the other given instance of supported type.

        Parameter
        ------
        other : Node, int, or float
            The object to compare with.

        Returns
        ------
        bool
            True if greater than the given object; and False, otherwise.
        """
        try:
            return self.var > other.var
        except AttributeError:
            return self.var > other


    def __le__(self, other):
        """Compares two objects if the instance has value less than or equal to the other given instance of supported type.

        Parameter
        ------
        other : Node, int, or float
            The object to compare with.

        Returns
        ------
        bool
            True if less than or equal to the given object; and False, otherwise.
        """
        try:
            return self.var <= other.var
        except AttributeError:
            return self.var <= other


    def __ge__(self, other):
        """Compares two objects if the instance has value greater than or equal to the other given instance of supported type.

        Parameter
        ------
        other : Node, int, or float
            The object to compare with.

        Returns
        ------
        bool
            True if grater than or equal to the given object; and False, otherwise.
        """
        try:
            return self.var >= other.var
        except AttributeError:
            return self.var >= other


    def __eq__(self, other):
        """Compares two objects if the instance has value equal to the other given Node object.

        Parameter
        ------
        other : Node
            The object to compare with.

        Returns
        ------
        bool
            True if equal to the given object; and False, otherwise.
        """
        try:
            return self.var == other.var
        except:
            raise TypeError('Input has incomparable type.')


    def __ne__(self, other):
        """Compares two objects if the instance has value unequal to the other given object.

        Parameter
        ------
        other : Node, int, float, or any type
            The object to compare with.

        Returns
        ------
        bool
            True if not equal to the given object; and False, otherwise.
        """
        return not self.__eq__(other)


    def __abs__(self):
        """Returns a new Node instance that has the absolute value.

        Returns
        ------
        Node
            a new Node instance that has the absolute value
        """
        new_abs = Node(abs(self.var))
        self.child.append((1, new_abs))
        return new_abs


    def __pow__(self, other):
        """Returns the exponential of the Node instance as base and another given instance of supported type as an exponent.

        Parameter
        ------
        other : Node, int, or float
            the exponent part.

        Returns
        ------
        Node
            the result of the exponential operation as a new Node instance.
        """
        try:
            new_val = Node(self.var ** other.var)
            self.child.append((new_val, (other.var) * self.var ** (other.var-1)))
            other.child.append((new_val, self.var ** other.var * (np.log(self.var))))
            return new_val
        except:
            if isinstance(other, int) or isinstance(other, float):
                new_val = Node(self.var ** other)
                self.child.append((new_val, (other) * self.var ** (other-1)))
                return new_val
            else:
                raise TypeError(f"Exponent is invalid.")


    def __rpow__(self, other):
        """Returns the reversed exponential of the Node instance as base and another given instance of supported type as an exponent.

        Parameter
        ------
        other : int, or float
            the base of the exponential function.

        Returns
        ------
        Node
            the result of the exponential operation as a new Node instance.
        """
        try:
            new_val = Node(other ** self.var)
        except:
            raise ValueError("must be a number.")
        self.child.append((new_val, other**self.var * np.log(other)))
        return new_val

        
    @staticmethod
    def log(var, base = None):
        """Logarithmic function supporting operations for reverse mode AD.

        Parameter
        ------
        var : Node, int or float
            value to compute the log
        base : Node, int or float
            base value of log function, optional (default = None assumed natural e)
        """
        try:
            if var.var <= 0:
                raise ValueError('Input must to be greater than 0.')
        except:
            raise TypeError(f"Invalid input type.")

        if base is None:
            log_var = Node(np.log(var.var))
            var.child.append((log_var, (1. / var.var) * 1))
            return log_var

        log_var = Node(np.log(var.var) / np.log(base.var))
        var.child.append((log_var, (1 / var.var / np.log(base.var)) * 1))
        return log_var
        

    @staticmethod
    def sqrt(var):
        """square root function supporting operations for reverse mode AD.
    
        Parameter
        ------
        var : Node, int or float
            value to compute square root
        """
        if var < 0:
            raise ValueError("Invalid input: value must be greater than or equal to zero.")
        else:
            try:
                sqrt_var = Node(var.var**(1/2))
                var.child.append((sqrt_var, (1/2)*var.var**(-1/2)))
            except:
                raise TypeError(f"Invalid input type.")
        return sqrt_var


    @staticmethod
    def exp(var):
        """exponential function (base natural) supporting operations for reverse mode AD.
    
        Parameter
        ------
        var : Node, int or float
            value to compute

        Notes
        ------
        exponential functions for other bases are handled by __pow__ in the Node class.
        """
        try:
            new_val = Node(np.exp(var.var))
            var.child.append((new_val, np.exp(var.var) * 1))
            return new_val
        except:
            if not isinstance(var, int) and not isinstance(var, float):
                raise TypeError(f"Invalid input type.")
        
            return np.exp(var)


    @staticmethod
    def sin(var):
        """Sine function supporting operations for reverse mode AD.

        Parameter
        ------
        var : Node, int or float
            value to compute sine
        """
        try:
            new_val = Node(np.sin(var.var))
            var.child.append((new_val, 1 * np.cos(var.var)))
            return new_val
        except:
            if not isinstance(var, int) and not isinstance(var, float):
                raise TypeError(f"Invalid input")
        
            return np.sin(var)


    @staticmethod
    def cos(var):
        """Cosine function supporting operations for reverse mode AD.

        Parameter
        ------
        var : Node, int or float
            value to compute cosine
        """
        try:
            new_val = Node(np.cos(var.var))
            var.child.append((new_val, 1 * -np.sin(var.var)))
            return new_val
        except:
            if not isinstance(var, int) and not isinstance(var, float):
                raise TypeError(f"Invalid input")
        
            return np.cos(var)
    
    
    @staticmethod
    def tan(var):
        """Tangent function supporting operations for reverse mode AD.

        Parameter
        ------
        var : Node, int or float
            value to compute tangent
        """
        try:
            new_val = Node(np.tan(var.var))
            var.child.append((new_val, 1 * 1 / np.power(np.cos(var.var), 2)))
            return new_val
        except:
            if not isinstance(var, int) and not isinstance(var, float):
                raise TypeError(f"Input {var} is not valid.")
        
            return np.tan(var)


    @staticmethod
    def arcsin(var):
        """Inverse sine function supporting operations for reverse mode AD.

        Parameter
        ------
        var : Node, int or float
            value to compute inverse sine
        """
        try:
            if var.var > 1 or var.var < -1:
                raise ValueError('Please input -1 <= x <=1')
            else:
                new_val = Node(np.arcsin(var.var))
                var.child.append((new_val, 1 / np.sqrt(1 - (var.var ** 2))))
                return new_val
        except:
            if not isinstance(var, int) and not isinstance(var, float):
                raise TypeError(f"Input {var} is not valid.")
            return np.arcsin(var)


    @staticmethod
    def arccos(var):
        """Inverse cosine function supporting operations for reverse mode AD.

        Parameter
        ------
        var : Node, int or float
            value to compute inverse cosine
        """
        try:
            if isinstance(var, int) or isinstance(var, float):
                return np.arccos(var)

            if var.var > 1 or var.var < -1:
                raise ValueError('Please input -1 <= x <=1')
            else:
                new_val = Node(np.arccos(var.var))
                var.child.append((new_val, -1 / np.sqrt(1 - (var.var ** 2))))
            return new_val
        except:
                raise TypeError(f"Input {var} is not valid.")


    @staticmethod
    def arctan(var):
        """Inverse tangent function supporting operations for reverse mode AD.

        Parameter
        ------
        var : Node, int or float
            value to compute inverse tangent
        """
        try:
            new_val = Node(np.arctan(var.var))
            var.child.append((new_val, 1 * 1 / (1 + np.power(var.var, 2))))

            return new_val

        except AttributeError:
            return np.arctan(var)


    @staticmethod
    def sinh(var):
        """Hyperbolic sine function supporting operations for reverse mode AD.

        Parameter
        ------
        var : Node, int or float
            value to compute inverse hyperbolic sine
        """
        try:
            new_val = Node(np.sinh(var.var))
            var.child.append((new_val, 1 * np.cosh(var.var)))
            return new_val

        except AttributeError:
            return np.sinh(var)


    @staticmethod
    def cosh(var):
        """Hyperbolic cosine function supporting operations for reverse mode AD.

        Parameter
        ------
        var : Node, int or float
            value to compute inverse hyperbolic cosine
        """
        try:
            new_val = Node(np.cosh(var.var))
            var.child.append((new_val, 1 * np.sinh(var.var)))

            return new_val

        except AttributeError:
            return np.cosh(var)


    @staticmethod
    def tanh(var):
        """Hyperbolic tangent function supporting operations for reverse mode AD.

        Parameter
        ------
        var : Node, int or float
            value to compute inverse hyperbolic tangent
        """
        try:
            new_val = Node(np.tanh(var.var))
            var.child.append((new_val, 1 * 1 / np.power(np.cosh(var.var), 2)))
            return new_val
        except AttributeError:
            return np.tanh(var)

    @staticmethod
    def logistic(var):
        """Logistic function supporting operations for reverse mode AD.

        Parameter
        ------
        var : Node, int or float
            value to compute the logistic
        """
        try:
            logistic_var = Node(1 / (1 + np.exp(-var.var)))
            var.child.append((logistic_var, 1 / (1 + np.exp(-var.var)) * (1-(1 / (1 + np.exp(-var.var)) * 1))))
            return logistic_var
        except:
            raise TypeError(f"Invalid input type.")   
        