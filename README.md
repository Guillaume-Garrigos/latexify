# latexify

Install with `pip install latexifier`

Use with `from latexifier import latexify`

This module aims to turn python objects into latex strings. For instance:

- Integers: `latexify(20)` returns `"20"`
- Python Fractions: `latexify(Fraction(1,3))` returns `"\frac{1}{3}"`
- Approximate Fractions: `latexify(0.25)` returns `"\frac{1}{4}"`
- Approximate combination of radicals: `latexify(0.5 - 3 * numpy.sqrt(5))` returns `"\frac{1}{2} - 3\sqrt{5}"`
- Complex numbers: `latexify(numpy.sqrt(2) + 3.5j)` returns `"\sqrt{2}+\frac{7}{2}i"`
- numpy.arrays: `latexify(numpy.eye(2))` returns `"\begin{pmatrix} 1&0\\0&1 \end{pmatrix}"` 
- tuples: `latexify((1,2,3))` returns `"\left(1,2,3\right)"`
- lists: `latexify([1,2,3])` returns `"1,2,3"`
- sympy polynomials: `latexify(x**2 - y*z/3)` returns `"x^2 - \frac{yz}{3}"`
- All those objects can be combined and nested: list of arrays, arrays of polynomials, etc.

When given integer-based values (such as Integers or Fractions), `latexify` returns a string without any ambiguity. 
When given a float, `latexify` tries to approximate this float by means of fractions and radicals. To so, the search is limited to "small" fractions and "small" radicals. This behavior can be changed with the function `latexifier.parameters(parameter_name=parameter_value)`. Here are a few important ones:

- denominator_max (default: 10). Maximal denominator allowed in fractions
- radical_max (defaut: 7). We allow for sqrt(n) to show up, with n <= radical_max
- frmt (default: '{:3.6f}'). format for rounding reals when no nice approximation is found
