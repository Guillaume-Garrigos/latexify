import numpy as np
import sympy
import numbers
import fractions
from fractions import Fraction
from latexifier.CFraction import CFraction
from mpmath import pslq

def printmk(*tuple_of_text):
    from IPython.display import display, Markdown
    L = [Markdown(text) for text in tuple_of_text]
    return display(*tuple(L))

def default_parameters():
    return {
            'denominator_max' : 10, # maximal denominator allowed in fractions
            'root_max' : 7, # We allow for sqrt(n) to show up, with n <= radical_max
            'style_fraction' : 'frac', # how to display fractions 
            'frmt' : '{:3.6f}', # format for rounding reals when no nice approximation is found
            'newline' : False, # should the latex string contain \n newline characters?
            'arraytype' : 'pmatrix', # arrays can be converted in many latex flavours
            'list_separator' : ', ', # sparator used between elements of a list
            'mathmode' : 'raw', # can be 'raw', 'inline' (gets in between $ $), 'equation' (gets inside a equation* environment, 'display' (gets in between \[ ... \])
            'tol' : 1e-12, # tolerance when approximating floats with algebraics
            'constants' : []
        }


LATEXIFY_DEFAULT_PARAM = default_parameters()

def get_parameters(**args):
    if args == {}:
        return LATEXIFY_DEFAULT_PARAM
    else:
        return {**LATEXIFY_DEFAULT_PARAM, **args}

def parameters(**args):
    global LATEXIFY_DEFAULT_PARAM
    LATEXIFY_DEFAULT_PARAM = get_parameters(**args)
    
from warnings import warn

def real_to_fraction_maybe(a, verbose=False, tol=1e-12, **param):
    # convert a real to the closest rational, with bounded denominator
    # we verify that the approximation is good (controled by 'tol')
    #   if it is not, we return the original real number
    param = get_parameters(**param)
    frac = Fraction(a).limit_denominator(param['denominator_max'])
    if np.abs(frac - a) < tol:
        return frac, True
    else:
        if verbose:
            warn("A real number couldn't properly be converted into a Fraction. The Fraction "+str(frac)+" is a bad approximation for "+str(a)+".")
        return a, False

def real_to_fraction_x_root_maybe(x, **param):
    # tries to convert a real to something like sqrt(a)*b/c
    # 'a' cannot be larger than 'radical_max'
    # Here is a stupid implementation
    param = get_parameters(**param)
    for n in range(1, param['radical_max']+1):
        y = x / np.sqrt(n) # y might be a fraction now
        frac, success = real_to_fraction_maybe(y, **param)
        if success:
            return frac * sympy.sqrt(n), True
    return x, False


def nonsquare_int(int_max=11):
    # returns a list of intergers < int_max with no square factors
    squares = [n**2 for n in np.arange(2,np.ceil(np.sqrt(int_max)))]
    sqdiv = []
    for k in range(int_max+1):
        sqdiv = sqdiv + [k*s for s in squares]
    L = [1]
    for n in range(2, int_max+1):
        if n not in sqdiv:
            L.append(n)
    return L

def real_to_simple_radical_maybe(x, **param):
    """ Tries to turn a float into a rational combination of roots.
    If it fails, returns directly the float with an error message.
    It would be nice to also take into account other constants like pi?
    """
    param = get_parameters(**param)
    list_roots = nonsquare_int(param['root_max'])
    L = [np.sqrt(n) for n in list_roots]
    L.append(x)
    coeff = pslq(L, tol=param['tol'])
    if coeff is None: # we found no algebraic combination
        return x, False
    denominator = -coeff[-1]
    if denominator == 0: # this shouldn't happen, but who knows..
        return x, False
    fracs = [Fraction(c,denominator) for c in coeff[:-1]]
    for frac in fracs:
        if frac.denominator > param['denominator_max']:
            return x, False
    approx = 0
    for k in range(len(fracs)):
        approx = approx + fracs[k]*sympy.sqrt(list_roots[k])
    return approx, True


def fraction_to_latex(x, **param):
    """ style_fraction can be 'frac', 'dfrac' or 'inline' """
    style_fraction = get_parameters(**param)['style_fraction']
    if x.denominator == 1:
        return str(x.numerator)
    if x.numerator == 0:
        return '0'
    if style_fraction == 'inline':
        return str(x)
    elif style_fraction in {'frac', 'dfrac'} :
        return '\\'+ style_fraction + '{' + str(x.numerator) + '}{' + str(x.denominator) + '}'
    else:
        raise ValueError("Unknown value for the option 'style_fraction'")
        

def simple_radical_to_latex(x, **param):
    return sympy.latex(x)
    

def number_to_latex(x, **param):
    param = get_parameters(**param)
    return number_to_algebraic(x, **param)[0]

def simplify_number(x, **param):
    param = get_parameters(**param)
    return number_to_algebraic(x, **param)[1]
        
def number_to_algebraic(x, **param):
    param = get_parameters(**param)
    approx = x
    latex = ''
    if isinstance(x, numbers.Number):
        if isinstance(x, numbers.Complex):
            if isinstance(x, numbers.Real):
                if isinstance(x, numbers.Rational):
                    if isinstance(x, numbers.Integral) or isinstance(x, sympy.numbers.Integer): # x is an integer TRICKY : https://stackoverflow.com/questions/48458438/why-is-numpy-int32-not-recognized-as-an-int-type
                        latex = str(x)
                    elif isinstance(x, Fraction): # x is a Fraction
                        latex = fraction_to_latex(x, **param)
                    elif isinstance(x, sympy.numbers.Rational): # x is a sympy Rational
                        approx = Fraction(x.p, x.q)
                        latex = fraction_to_latex(approx, **param)
                    else:
                        raise ValueError('You gave me a Rational which is neither integer or Fraction')
                else: # x is complicated to deal with
                    x = float(x) # just to prevent annoying stuff like sympy floats
                    # maybe it is a simple fraction in disguise?
                    frac, success = real_to_fraction_maybe(x, **param)
                    if success: # yay
                        latex = fraction_to_latex(frac, **param)
                        approx = frac
                    else:
                        # maybe it is a combination of fraction and square root
                        algebraic, success = real_to_simple_radical_maybe(x, **param)
                        if success:
                            latex = simple_radical_to_latex(algebraic, **param) 
                            approx = algebraic
                        else: # well we can just round it now              
                            latex = param['frmt'].format(x) # need to shorten useless zeros
                            approx = float(latex)
                            warn("A float got rounded")
            else: # we have a complex number to deal with
                latex_real, approx_real = number_to_algebraic(x.real, **param)
                latex_imag, approx_imag = number_to_algebraic(x.imag, **param)
                latex = latex_real + ' + ' + latex_imag + ' i'
                approx = CFraction(approx_real, approx_imag)
        else:
            raise ValueError('You gave me a number which is not a Complex')
    else:
        raise ValueError('You gave me something which is not a number')
    
    return latex, approx
    

    
def latexifier(x, **param):
    param = get_parameters(**param)
    if isinstance(x, list):
        latex = list_to_latex(x, **param)
    elif isinstance(x, tuple):
        latex = tuple_to_latex(x, **param)
    elif isinstance(x, np.ndarray):
        latex = numpyarray_to_latex(x, column=True, **param)
    elif isinstance(x, numbers.Number):
        latex = number_to_latex(x, **param)
    elif isinstance(x, sympy.Basic):
        latex = sympy_polynomial_to_latex(x, **param)
    else:
        raise ValueError('Unknown data type. Can be: numbers, arrays, lists')
        
    return latex

def latexify(x, verbose=False, **param):
    param = get_parameters(**param)
    latex = latexifier(x, **param)
    
    if param['newline']:
        end = '\n'
    else:
        end = ''
    
    if   param['mathmode'] == 'inline':
        latex = '$'+latex+'$'
    elif param['mathmode'] == 'equation':
        latex = r'\begin{equation*} ' + end + latex + end + r' \end{equation*}'
    elif param['mathmode'] == 'display':
        latex = '\\[ ' + end + latex + end + ' \\]'
        
    if verbose == True:
        print(latex)
    elif verbose in {'printmk', 'markdown'}:
        if param['mathmode'] == 'raw':
            printmk('$'+latex+'$')
        else:
            printmk(latex)
            
    return latex


def sympy_polynomial_to_latex(poly, **param):
    """ In sympy there is a bunch of notions of 'polynomials'
        E = x**2 + y**2 - 2*x*y is an 'expression' of type 'sympy.core.add.Add'
        P = sympy.poly(E, x, y) is a 'polynomial' of type 'sympy.polys.polytools.Poly'
            (and can be covert back with P.as_expr())
        There is also some structures that we obtain as a derivative of polynomials
        For instance P.factor() looks like an 'expression' but is of type 'sympy.core.mul.Mul' ...
        So it is a mess for me, but luckily there is sympy.latex() which is able to quite nicely convert everything into latex strings
        
        https://stackoverflow.com/questions/62324998/construct-sympy-poly-from-coefficients-and-monomials
    """
    if isinstance(poly, sympy.polys.polytools.Poly):
        # here we can do intersting stuff like replacing floats with Fractions
        dico = poly.as_dict()
        polyfrac = sympy.Poly.from_dict({ key : simplify_number(dico[key], **param) for key in dico.keys() }, poly.gens)
        print([simplify_number(dico[key], **param) for key in dico.keys() ])
        latex = sympy.latex(polyfrac.as_expr())
    else: 
        try: # if it is an expression we can try to convert it to a polynomial
            proper_poly = sympy.Poly(poly)
        except: # if it fails we just let sympy do something
            latex = sympy.latex(poly)
        else: # if it succeeds we latexify the polynomial
            latex = sympy_polynomial_to_latex(proper_poly, **param)
            
    return latex.replace(' ','')



######################################
# Structures


def numpyarray_to_latex(a, column=True, **param):
# taken from https://github.com/josephcslater/array_to_latex
    param = get_parameters(**param)
    if len(a.shape) > 2:
        raise ValueError('You gave me an array which has more than two dimensions : I cannot turn it into a latex array')
    if len(a.shape) == 1:
        a = np.array([a])
        if column:
            a = a.T
            
    if param['newline']:
        end = '\n'
    else:
        end = ''
        
    out = r'\begin{' + param['arraytype'] + '} ' + end
    for i in np.arange(a.shape[0]):
        for j in np.arange(a.shape[1]):
            out = out + latexifier(a[i,j], **param) + r' & '
        out = out[:-2]
        out = out + '\\\\ ' + end
    out = out[:-3] + end + r'\end{' + param['arraytype'] + '}'
    return out


def list_to_latex(L, **param):
    """ returns the elements of the list one next to each other """
    param = get_parameters(**param)
    latex = ''
    for x in L:
        latex = latex + latexifier(x, **param) + param['list_separator']
    return latex[:-len(param['list_separator'])]  

def tuple_to_latex(tup, **param):
    """ returns the elements of the tuple one next to each other, in the form (..,..,..) """
    param = get_parameters(**param)
    latex = '\\left('
    for x in tup:
        latex = latex + latexifier(x, **param)+', '
    latex = latex[:-2] + ' \\right)'
    return latex    