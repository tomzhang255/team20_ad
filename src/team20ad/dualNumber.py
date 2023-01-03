import numpy as np


class DualNumber:
    """A dual number class supporting operations for forward mode automatic differentation.
    
    Attributes
    ------
    _supported_scalars : tuple
        A tuple containing types of objects that are supported by the
        dual number operations.
    real : int or float
        The real part of a dual number, which represents the value of user
        defined function(s) 'f' evaluated at point 'x'.
    dual : int or float
        The dual part of a dual number, corresponding to the derivative
        of user defined functions(s) 'f' evaluated at point 'x'.

    Examples
    ------
    >>> DualNumber(3.0, 4)
    DualNumber(3.0, 4)
    >>> DualNumber(3)
    DualNumber(3, 1.0)
    """

    _supported_scalars = (int, float)

    def __init__(self, real, dual = 1.0):
        """
        Parameters
        ------
        real : int or float
            The value of user defined function(s) 'f' evaluated at point 'x'.
        dual : int or float, optional (default = 1.0)
            The corresponding derivative of user defined functions(s) 'f' evaluated at point 'x'.
        
        Raises
        ------
        TypeError
            if an argument value is of unsupported type. 
        """
        if isinstance(real, self._supported_scalars):
            self.real = real
            self.dual = dual
        else:
            raise TypeError("Supported scalars: {_supported_scalars}")

    def __repr__(self):
        """Returns a representation of the DualNumber instance.

        Returns
        ------
        str
            a representation of the DualNumber instance.
        """
        return f"DualNumber({self.real}, {self.dual})"

    def __str__(self):
        """Returns a string representation of the DualNumber instance.

        Returns
        ------
        str
            a string representation of the DualNumber instance.
        """
        return f"DualNumber: real = {self.real}, dual = {self.dual}"

    def __neg__(self):
        """Returns the negation of the DualNumber instance.

        Returns
        ------
        DualNumber
            the negation of the DualNumber instance for both real and dual parts.
        """
        return DualNumber(-self.real, -self.dual)

    def __add__(self, other):
        """Returns the sum of the DualNumber instance and another given instance of supported type.

        Parameter
        ------
        other : DualNumber, int, or float
            the instance to compute the sum with.

        Returns
        ------
        DualNumber
            the sum of the two instances.
        """
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real + other, self.dual)
        return DualNumber(self.real + other.real, self.dual + other.dual)

    def __radd__(self, other):
        """Returns the sum of the DualNumber instance and another given instance of supported type.

        As the add operation is commutative, this method delegates the operation
        to __add__().

        Parameter
        ------
        other : int, or float
            a scalar object to compute the sum with.

        Returns
        ------
        DualNumber
            the sum of the two instances.
        """
        return self.__add__(other)

    def __sub__(self, other):
        """Returns the substraction of the DualNumber instance and another given instance of supported type.
        
        Parameter
        ------
        other : DualNumber, int, or float
            the instance to subtract.

        Returns
        ------
        DualNumber
            the substraction of the two instances.
        """
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real - other, self.dual)
        return DualNumber(self.real - other.real, self.dual - other.dual)

    def __rsub__(self, other):
        """Returns the substraction of a scalar object and a DualNumber

        Parameter
        ------
        other : int, or float
            a scalar object to be subtracted.

        Returns
        ------
        DualNumber
            the substraction of the two instances.
        """
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            return DualNumber(other - self.real , -self.dual)
        return DualNumber(other.real - self.real, other.dual - self.dual)

    def __mul__(self, other):
        """Returns the product of the DualNumber instance and another given instance of supported type.

        Parameter
        ------
        other : DualNumber, int, or float
            the instance to compute the product with.

        Returns
        ------
        DualNumber
            the product of the two instances.
        """
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            return DualNumber(self.real * other, other * self.dual)
        return DualNumber(self.real * other.real,
                          self.real * other.dual + other.real * self.dual)

    def __rmul__(self, other):
        """Returns the product of the DualNumber instance and another scalar object.

        As the multiplication operation is commutative, this method delegates the operation
        to __mul__().

        Parameter
        ------
        other : int, or float
            a scalar object to compute the product with.

        Returns
        ------
        DualNumber
            the product of the two instances.
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """Returns the division of the DualNumber instance and another given instance of supported type.

        Parameter
        ------
        other : DualNumber, int, or float
            the instance to divide.

        Returns
        ------
        DualNumber
            the division of the two instances.
        """
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            return DualNumber(self.real / other, self.dual / other)
        if other.real == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return DualNumber(self.real / other.real,
                          (self.dual * other.real - self.real * other.dual) / (other.real ** 2))

    def __rtruediv__(self, other):
        """Returns the division of the DualNumber instance and a scalar object.

        Parameter
        ------
        other : DualNumber, int, or float
            the instance to be divided from.

        Returns
        ------
        DualNumber
            the division of the two instances.
        """
        if not isinstance(other, self._supported_scalars):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if other == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return DualNumber(other / self.real, (-other / self.real ** 2) * self.dual)

    def __pow__(self, other):
        """Returns the exponential of the DualNumber instance as base and another given instance of supported type as an exponent.

        Parameter
        ------
        other : DualNumber, int, or float
            the exponent part.

        Returns
        ------
        DualNumber
            the result of the exponential operation on the given instances.
        """
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            real_pow = self.real ** other
            dual_pow = other * (self.real ** (other - 1)) * self.dual
        else:
            if self.real > 0:
                real_pow = self.real ** other.real
                dual_pow = other.real * self.real ** (other.real - 1) * self.dual + np.log(
                    self.real) * self.real ** other.real * other.dual
            else:
                real_pow = self.real ** other.real
                dual_pow = other.real * \
                    self.real ** (other.real - 1) * self.dual
        return DualNumber(real_pow, dual_pow)

    def __rpow__(self, other):
        """Returns the exponential of the DualNumber instance as an exponent and another given instance of supported type as base.

        Parameter
        ------
        other : DualNumber, int, or float
            the base of the exponential function.

        Returns
        ------
        DualNumber
            the result of the exponential operation on the given instances.
        """
        if not isinstance(other, self._supported_scalars):
            raise TypeError(f"Unsupported type '{type(other)}'")
        real_pow = other ** self.real
        if other < 0:
            raise ValueError(f"Unsupported value '{type(other)}'")
        dual_pow = np.log(other) * other ** self.real * self.dual
        return DualNumber(real_pow, dual_pow)

    def __eq__(self, other):
        """Compares two objects if they are equal.

        Parameter
        ------
        other : DualNumber, int, or float
            The object to compare with.

        Returns
        ------
        bool
            True if the two instances are equal; and False, otherwise.
        """
        if not isinstance(other, (*self._supported_scalars, DualNumber)):
            raise TypeError(f"Unsupported type '{type(other)}'")
        if isinstance(other, self._supported_scalars):
            return (self.real == other.real) and (self.dual == 0)
        return (self.real == other.real) and (self.dual == other.dual)

    def __ne__(self, other):
        """Compares two objects if they are not equal.

        This operator is a negation of __eq__(), and hence delegates the task to
        __eq__().

        Parameter
        ------
        other : DualNumber, int, or float
            The object to compare with.

        Returns
        ------
        bool
            True if the two instances are not equal; and False, otherwise.
        """
        return not self.__eq__(other)

    def __lt__(self, other):
        """Compares two objects if the instance is less than the other given instance of supported type.

        Parameter
        ------
        other : DualNumber, int, or float
            The object to compare with.

        Returns
        ------
        bool
            True if less than the given object; and False, otherwise.
        """
        try:
            return self.real < other.real
        except AttributeError:
            return self.real < other

    def __gt__(self, other):
        """Compares two objects if the instance is greater than the other given instance of supported type.

        Parameter
        ------
        other : DualNumber, int, or float
            The object to compare with.

        Returns
        ------
        bool
            True if greater than the given object; and False, otherwise.
        """
        try:
            return self.real > other.real
        except AttributeError:
            return self.real > other

    def __le__(self, other):
        """Compares two objects if the instance is less than or equal to the other given instance of supported type.

        Parameter
        ------
        other : DualNumber, int, or float
            The object to compare with.

        Returns
        ------
        bool
            True if less than or equal to the given object; and False, otherwise.
        """
        try:
            return self.real <= other.real
        except AttributeError:
            return self.real <= other

    def __ge__(self, other):
        """Compares two objects if the instance is greater than or equal to the other given instance of supported type.

        Parameter
        ------
        other : DualNumber, int, or float
            The object to compare with.

        Returns
        ------
        bool
            True if greater than or equal to the given object; and False, otherwise.
        """
        try:
            return self.real >= other.real
        except AttributeError:
            return self.real >= other

    def __abs__(self):
        """Returns the absolute value of the DualNumber instance.

        Returns
        ------
        DualNumber
            the absolute value of the DualNumber instance on both real and dual parts.
        """
        return DualNumber(abs(self.real), abs(self.dual))