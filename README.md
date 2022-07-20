# latexify

Install with `pip install latexifier`

Use with `from latexifier import latexify`

This module aims to turn python objects into latex strings. For instance:

- Integers: `latexify(20)` returns `"20"`
- Python Fractions: `latexify(Fraction(1,3))` returns `"\frac{1}{3}"`
- Approximate Fractions: `latexify(0.25)` returns `"\frac{1}{4}"`
- Approximate combination of radicals: `latexify(0.5 - 3 * numpy.sqrt(5), numbertype='algebraic')` returns `"\frac{1}{2} - 3\sqrt{5}"`
- Complex numbers: `latexify(2 + 3.5j)` returns `"2+\frac{7}{2}i"`
- numpy.arrays: `latexify(numpy.eye(2))` returns `"\begin{pmatrix} 1&0\\0&1 \end{pmatrix}"` 
- tuples: `latexify((1,2,3))` returns `"\left(1,2,3\right)"`
- lists: `latexify([1,2,3])` returns `"1,2,3"`
- sympy polynomials: `latexify(x**2 - y*z/3)` returns `"x^2 - \frac{yz}{3}"`
- All those objects can be combined and nested: list of arrays, arrays of polynomials, etc.

When given integer-based values (such as Integers or Fractions), `latexify` returns a string without any ambiguity. 
When given a float, `latexify` tries to approximate this float by means of fractions and radicals. To do so, the search is limited to "small" fractions and "small" radicals. What "small" means can be specified with some parameters (see below).

### Options

The behavior of latexify can be changed *on-the-fly* by passing arguments to the function, like `latexify(0.25, parameter_name=parameter_value)`, or can be set *once-for-all* with the function `latexifier.parameters(parameter_name=parameter_value)`. Here are is a list of such parameters:

| parameter_name | default value | purpose |
| -------------- | :-: | ------- |
| style_fraction | 'frac'  | Sets how fraction should be displayed. Given `0.5`, the option 'frac' returns `\frac{1}{2}`. The option 'dfrac' returns `\dfrac{1}{2}`. The option 'inline' returns `1/2`. |
| newline | False | Some Latex expressions usually contain newlines (for arrays for instance). By default latexify returns a string with no such newlines, but you can turn it on. | 
| arraytype | 'pmatrix' | np.arrays can be converted in many latex flavours, of the form `\begin{arraytype} ... \end{arraytype}` |
| column | True | if the input is a 1D array, how should it be represented? 'True' means a column vector, 'False' means a row vector |
| mathmode | 'raw' | latexify returns by default the latex expression of a number or a matrix, etc. If you want to display it correctly in a document, this string must be placed in a math environment. You can optionally ask latexify to do it for you. With the option 'inline', the expression is returned in between `$ $`. With the option 'equation'  the expression is returned inside a `equation*` environment. With the option 'display' the expression is returned in between `\[ ... \]`. |
| denominator_max | 10           | Maximal denominator allowed to appear in fractions (in absolute value) |
| numbertype | 'rational' | Controls which kind of number you expect to have. By default latexify looks only for formal expression of rationals. With the option 'root' it will look for a rational multiplied with the square root of some integer (see `root_max` below). With the option 'algebraic' it will look for a rational linear combination of square roots. Be careful though, these two options are more time consuming. |
| root_max | 7 | The expression can contain sqrt(n) to show up, with n no greater than root_max |
| tol | 1e-12 | Any formal expression must satisfy \vert x_{formal} - x_{float} \vert < tol. If not, the float is rounded.  |
| frmt | '{:3.6f}' | When no formal expression can be found, the floats will be rounded. This parameters specifies which rounding/formating rules must be applied (this is a [pythonic](https://pyformat.info/) syntax). |
| verbose | False | When calling latexify we can ask it to print the result. Can be 'True' (the string is printed), 'False' (nothing is printed), 'markdown' (in notebooks this interprets the latex contents of the string and renders the expression like in markdown) |
| value | False | if 'True', latexify returns a second variable which is the approximation of the input  |
