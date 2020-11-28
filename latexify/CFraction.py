# taken from https://github.com/gladstonedigital/mathvis/blob/master/mathvis/cfractions.py

"""Implementation of complex numbers (a+b*j) where the real and imaginary components \
are represented as Fraction instances to preserve accuracy during most mathematical \
operations.

When accuracy is lost due to conversion to float in order to perform an operation, \
the result is converted back into a CFraction instance before returning. Of course \
this doesn't recover any accuracy but the type stays the same.

Accuracy is not preserved during the following computations:
    - __abs__(), magnitude implemented as sqrt(a**2 + b**2)
    - __complex__(), converting to a built-in complex type
    - math operations involving a CFraction and an irrational number
        (becomes float during calculation)
    - math operations involving a CFraction and most floating point numbers
        certain floats are able to be converted to Fractions correctly but most aren't
    - CFraction raised to a non-integer power

Aside from the use of the Fraction class to store values, this class attempts to \
behave as closely as possible to the built-in complex class.
"""

import copy
import math
import operator
from collections.abc import Iterable
from fractions import Fraction
from functools import reduce
from numbers import Complex, Rational, Real

class _Fraction(Fraction):
    """Extend Fraction to override __repr__, to match functionality of complex() in the interpreter"""

    def __repr__(self):
        return self.__str__()

class CFraction(Complex):
    """CFraction(real[, imag]) -> complex number with components stored as Fraction instances.

    Create a complex number from a real part and an optional imaginary part. CFraction is interoperable with the built-in complex class, but of course this loses the benefit of Fraction components.
    """

    def __init__(self, real=0, imag=0):
        """Coerce real and imaginary components to fractions"""
        if isinstance(real, Complex) and imag == 0:
            real, imag = (real.real, real.imag)
        # Support passing tuples as (numerator,denominator) pairs
        self._real = _Fraction(*real) if isinstance(real, Iterable) and len(real) == 2 else _Fraction(real)
        self._imag = _Fraction(*imag) if isinstance(imag, Iterable) and len(imag) == 2 else _Fraction(imag)

# Properties
    @property
    def real(self):
        """Real component of complex number"""
        return self._real

    @property
    def imag(self):
        """Imaginary component of complex number"""
        return self._imag

# Methods
    def conjugate(self):
        """Return complex conjugate (negated imaginary component)"""
        return CFraction(self.real, -1 * self.imag)

    def limit_denominator(self, max_denominator=1000000):
        """Limit length of fraction at the cost of some accuracy"""
        return CFraction(self.real.limit_denominator(max_denominator),
                         self.imag.limit_denominator(max_denominator))

# Comparison operators
    def __eq__(self, other):
        """Check equality with CFractions or other types"""
        if other is None:
            return False

        if isinstance(other, Real):
            return self.imag == 0 and self.real == other
        return self.imag == other.imag and self.real == other.real

    def __ne__(self, other):
        return not self.__eq__(other)

# Unary operators
    def __abs__(self):
        """Return magnitude of complex number sqrt(a**2 + b**2)"""
        return _Fraction(math.sqrt(self.real**2 + self.imag**2))

    def __neg__(self):
        return CFraction(-1 * self.real, -1 * self.imag)

    def __pos__(self):
        return CFraction(self.real, self.imag)

    def __hash__(self):
        """Lifted this algorithm from implementation of built-in complex().__hash__ in complex_hash(PyComplexObject*) in Objects/complexobject.c"""
        hashreal = hash(self.real)
        hashimag = hash(self.imag)

        if hashreal == -1 or hashimag == -1:
            return -1

        combined = hashreal + 1000003 * hashimag
        return -2 if combined == -1 else combined

    def __reduce__(self):
        """Support for pickle"""
        return (self.__class__, (self.real, self.imag))

    def __copy__(self):
        """Support for copy module"""
        return self.__class__(self.real, self.imag)

    def __deepcopy__(self, memo):
        """Support for copy module"""
        return self.__class__(copy.deepcopy(self.real, memo), copy.deepcopy(self.imag, memo))

# Binary operators
    def __add__(self, other):
        return CFraction(self.real + other.real, self.imag + other.imag)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-1 * other)

    def __rsub__(self, other):
        return (-1 * self).__add__(other)

    def __mul__(self, other):
        return CFraction(self.real * other.real - self.imag * other.imag, self.real * other.imag + self.imag * other.real)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if other == 0:
            raise ZeroDivisionError("complex division by zero")

        return CFraction((self.real * other.real + self.imag * other.imag, other.real**2 + other.imag**2),
                         (self.imag * other.real - self.real * other.imag, other.real**2 + other.imag**2))

    def __rtruediv__(self, other):
        return CFraction(other).__truediv__(self)

    def __div__(self, other):
        return self.__truediv__(other)

    def __rdiv__(self, other):
        return CFraction(other).__truediv__(self)

    def __floordiv__(self, other):
        raise TypeError("can't take floor of complex number.")

    def __rfloordiv__(self, other):
        raise TypeError("can't take floor of complex number.")

    def __divmod__(self, other):
        raise TypeError("can't take floor or mod of complex number.")

    def __rdivmod__(self, other):
        raise TypeError("can't take floor or mod of complex number.")

    def __pow__(a, power):
        """Raise CFraction to power 'power'. 'power' can be Rational, CFraction, or other"""

        if a == 0 and (power.imag != 0 or power.real < 0):
            raise ZeroDivisionError("0 to a negative or complex power")

        if isinstance(power, Rational): # Rational(Real) exponents
            if power.denominator == 1: # integer exponents
                # I think Fraction always stores the sign in the numerator, but I'm not 100% sure.
                # this compensates just in case the numerator and denominator can vary in sign
                power = abs(power.numerator) if power >= 0 else -1 * abs(power.numerator)
                if power >= 0: # for positive and zero powers, just multiply repeatedly
                    return reduce(operator.mul, (a for _ in range(power)), CFraction(1))
                elif power < 0: # for negative exponents, invert fraction then raise to positive power
                    rn, rd = (a.real.numerator, a.real.denominator)
                    jn, jd = (a.imag.numerator, a.imag.denominator)
                    new_d = rn**2 * jd**2 + rd**2 * jn**2
                    return CFraction((rd * rn * jd**2, new_d), (-1 * rd**2 * jn * jd, new_d))**abs(power)

            else: # non-integer exponents use this https://stackoverflow.com/questions/3099403/calculating-complex-numbers-with-rational-exponents
                theta = math.atan2(a.imag, a.real)
                return CFraction(abs(a)**power * math.cos(power*theta), abs(a)**power * math.sin(power*theta))

        elif isinstance(power, Complex):
            if power.imag == 0 and isinstance(power.real, Rational) and a.real >= 0: # Complex power but actually real number
                return a**power.real

            # use built-in complex power code for complex or irrational powers
            return CFraction(complex(a)**complex(power))

        else:
            raise TypeError("unsupported operand type(s) for ** or pow(): '{}' and '{}'".format(a.__class__.__name__, power.__class__.__name__))

    def __rpow__(power, a):
        return CFraction(a).__pow__(power)

# Conversions
    def __str__(self):
        """(a+bj) or (a-bj) if nonzero real component else bj"""
        if self.real == 0:
            return str(self.imag) + "j"
        return "(" + str(self.real) + ("+" if self.imag >= 0 else "-") + str(abs(self.imag)) + "j)"

    def __repr__(self):
        return self.__str__()

    def __complex__(self):
        """Convert to built-in complex type"""
        return complex(self.real, self.imag)

    def __float__(self):
        raise TypeError("can't convert CFraction to float")

    def __int__(self):
        raise TypeError("can't convert CFraction to int")

