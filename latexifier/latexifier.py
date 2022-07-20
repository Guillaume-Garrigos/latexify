import numpy as np
import sympy
import numbers
import fractions
from fractions import Fraction
from mpmath import pslq
from warnings import warn

def printmk(*tuple_of_text):
    from IPython.display import display, Markdown
    L = [Markdown(text) for text in tuple_of_text]
    return display(*tuple(L))

def default_parameters():
    return {
            'numbertype' : 'rational', # can be 'root', 'algebraic'. 'algebraic' is quite expensive, 'root' is in between.
            'denominator_max' : 10, # maximal denominator allowed in fractions
            'root_max' : 7, # We allow for sqrt(n) to show up, with n <= root_max
            'frmt' : '{:3.6f}', # format for rounding reals when no nice approximation is found
            'style_fraction' : 'frac', # how to display fractions 
            'newline' : False, # should the latex string contain \n newline characters?
            'arraytype' : 'pmatrix', # arrays can be converted in many latex flavours
            'column' : True, # if the input is a 1D array, how should it be represented? 'True' means a column vector, 'False' means a row vector
            'list_separator' : ', ', # sparator used between elements of a list
            'mathmode' : 'raw', # can be 'raw', 'inline' (gets in between $ $), 'equation' (gets inside a equation* environment, 'display' (gets in between \[ ... \])
            'tol' : 1e-12, # tolerance when approximating floats with algebraics
            'constants' : [],
            'verbose' : False, # when calling latexify we can ask it to print the result. Can be 'True' (the string is printed), 'False' (nothing is printed), 'markdown' (in notebooks this interprets the latex contents of the string and renders the expression like in markdown),
            'value' : False, # if 'True', latexify returns a second variable which is the approximation of the input 
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

    
    
#-------------------------------------------------------
# Converting floats into Fractions
#-------------------------------------------------------

def real_to_fraction_maybe(a, **param):
    # convert a real to the closest rational, with bounded denominator
    # we verify that the approximation is good (controled by 'tol')
    # if it is not, we return the original real number
    # success is a bool telling if the approximation is faithful
    tol = param['tol']
    frac = Fraction(a).limit_denominator(param['denominator_max'])
    if np.abs(frac - a) < tol:
        approx = frac
        success = True
    else:
        approx = a
        success = False
    return approx, success

def display_fraction(numerator, denominator, **param):
    # numerator and denominator here are strings
    style_fraction = param['style_fraction']
    if denominator == '1':
        return numerator
    elif style_fraction == 'inline':
        return numerator + '/' + denominator
    elif style_fraction in {'frac', 'dfrac'} :
        return '\\'+ style_fraction + '{' + numerator + '}{' + denominator + '}'
    else:
        raise ValueError("Unknown value for the option 'style_fraction'")
    
def fraction_to_latex(x, **param):
    # x is supposed to be a number.Fraction here
    if x.denominator == 1:
        return str(x.numerator)
    elif x.numerator == 0:
        return '0'
    else:
        return display_fraction(str(x.numerator), str(x.denominator), **param)

#-------------------------------------------------------
# Converting floats into algebraic numbers
#-------------------------------------------------------
        
def nonsquare_int(int_max=11):
    # returns a list of 1 <= intergers <= int_max with no square factors
    squares = [n**2 for n in np.arange(2,np.ceil(np.sqrt(int_max)))]
    sqdiv = []
    for k in range(int_max+1):
        sqdiv = sqdiv + [k*s for s in squares]
    L = [1]
    for n in range(2, int_max+1):
        if n not in sqdiv:
            L.append(n)
    return L

def real_to_fraction_x_root_maybe(x, **param):
    # tries to convert a real to something like sqrt(a)*b/c
    # 'a' cannot be larger than 'root_max'
    # we return a sympy number, a certificate, and a latex string
    # Here is a stupid implementation
    for n in nonsquare_int(param['root_max']):
        y = x / np.sqrt(n) # y might be a fraction now
        frac, success = real_to_fraction_maybe(y, **param)
        if success:
            latex = fraction_x_root_to_latex((0,frac,n), **param)
            return frac * sympy.sqrt(n), True, latex
    return x, False, ''

def fraction_x_root_to_latex(x, **param):
    ''' 
        x is a triplet (a,b,c) with x = a+b*sqrt(c)
        a and b are Fraction, c is integer
    '''
    a = x[0]
    b = x[1]
    c = x[2]
    latexc = '\\sqrt{' + str(c) + '}'
    # we want to treat nicely the cases a=0, b=1, b<0, c=0, c=1
    # omg it became a total mess. it is a shame but works.
    if c*b == 0: # x = a
        if a == 0: # x=0
            return '0'
        else: # x = a
            return fraction_to_latex(a, **param)
    if c == 1: # x = a+b
        return fraction_to_latex(a+b, **param)
    # now we have square roots to deal with
    if b == 1: 
        if a == 0: # x = sqrt(c)
            return latexc
        else: # x = a + sqrt(c)
            return fraction_to_latex(a, **param) + '+' + latexc
    if b == -1:
        if a == 0: # x = -sqrt(c)
            return '-'+latexc
        else: # x = a - sqrt(c)
            return fraction_to_latex(a, **param) + '-' + latexc
    # now we have a square root times a fraction
    if b > 0 or a == 0: 
        denominator = str(b.denominator)
        if b.numerator == 1:
            numerator = latexc
        elif b.numerator == -1:
            numerator = '-' + latexc            
        else: 
            numerator = str(b.numerator) + latexc
        if a == 0: # x = b*sqrt(c)
            return display_fraction(numerator, denominator, **param)
        else: # x = a + b*sqrt(c)
            return fraction_to_latex(a, **param) + '+' + display_fraction(numerator, denominator, **param)
    else: # b<0 and a!=0 we need a - sign
        denominator = str(b.denominator)
        if abs(b.numerator) == 1:
            numerator = latexc
        else: 
            numerator = str(abs(b.numerator)) + latexc
        return fraction_to_latex(a, **param) + '-' + display_fraction(numerator, denominator, **param)


def real_to_simple_algebraic_maybe(x, **param):
    ''' Tries to turn a float into a rational combination of roots.
        If it succeeds, returns a sympy number
        If it fails, returns directly the float
        We return a number, a certificate, and a latex string
        It would be nice to also take into account other constants like pi?
        NB : this is very expensive. Can it be improved?
    '''
    list_roots = nonsquare_int(param['root_max'])
    L = [np.sqrt(n) for n in list_roots]
    L.append(x)
    coeff = pslq(L, tol=param['tol']) # TOO EXPENSIVE
    #coeff = [1,2,3,5,6,7,-1]
    if coeff is None: # we found no algebraic combination
        return x, False, ''
    denominator = -coeff[-1]
    if denominator == 0: # this shouldn't happen, but who knows..
        return x, False, ''
    fracs = [Fraction(c,denominator) for c in coeff[:-1]]
    for frac in fracs:
        if frac.denominator > param['denominator_max']:
            return x, False, ''
    approx = 0
    latex = ''
    for k in range(len(fracs)):
        if fracs[k] != 0:
            approx = approx + fracs[k]*sympy.sqrt(list_roots[k]) # it is expensive to use sympy.sqrt. Maybe hard code it?
            latex = latex + fraction_x_root_to_latex((0,fracs[k],list_roots[k]), **param) + '+'
    return approx, True, latex[:-1]

"""
def simple_algebraic_to_latex(x, **param):
    return sympy.latex(x)
"""


"""
def real_to_simple_radical_maybe(x, **param):
    # tries to convert a real to something like a+b*sqrt(n)
    # where a and b are Fractions with bounded denominator
    # 'n' cannot be larger than 'radical_max'
    # Here is a stupid implementation, surely we can do better
    #
    #now we set a=0
    param = get_parameters(**param)
    frac, success = real_to_fraction_maybe(x, **param)
    if success:
        return frac, True
    # now we know that x = a+b*sqrt(n) with b != 0, n!=0
    # the numerator for a is in the worst case x*denominator_max
    # the numerator for b is in the worst case x*denominator_max/sqrt(n)
    '''
    numerator_max = int(np.ceil(x * param['denominator_max']))
    for i in range(-numerator_max, numerator_max+1):
        for j in range(1, param['denominator_max']+1):
            radical, success = real_to_fraction_x_root_maybe(x - Fraction(i,j), **param)
            if success:
                return Fraction(i,j) + radical, True
    '''
    x, success = real_to_fraction_x_root_maybe(x, **param)
    return x, success
    #return x, False
"""

        
#-------------------------------------------------------
# Meta fucntions to deal with numbers
#-------------------------------------------------------    
        
        
def number_to_algebraic(x, **param):
    """ Input: x, typically a "number" (can mean many things)
        Output: latex, approx
            approx is a hopefully faithful simple equivalent formulation of x
                could be Fraction, or sympy number, polynomial etc
            latex is a string representing approx
        
        approx is required to be equal to x, up to the 'tol' parameter
        if the approximation fails, then we just return a short decimal approximation of x
    """
    approx = x
    latex = ''
    if isinstance(x, numbers.Number):
        if isinstance(x, numbers.Complex):
            if isinstance(x, numbers.Real):
                
                # FIRST : we check if x is already an algebraic structure that we can immediately recognize
                if isinstance(x, numbers.Rational):
                    if isinstance(x, numbers.Integral) or isinstance(x, sympy.numbers.Integer): 
                        # x is an integer 
                        # tricky : https://stackoverflow.com/questions/48458438/why-is-numpy-int32-not-recognized-as-an-int-type
                        latex = str(x)
                    elif isinstance(x, Fraction): # x is a Fraction
                        latex = fraction_to_latex(x, **param)
                    elif isinstance(x, sympy.numbers.Rational): # x is a sympy Rational
                        approx = Fraction(x.p, x.q)
                        latex = fraction_to_latex(approx, **param)
                    else: # x is a Rational but not a type we recognize --> error
                        raise ValueError('You gave me a Rational which is neither integer or Fraction')
                        
                # SECOND : x is complicated to deal with : it is not obviously an algebraic structure
                # we will need to do approximations
                else: 
                    x = float(x) # just to prevent annoying stuff like sympy floats
                    # maybe it is a simple fraction in disguise?
                    frac, success = real_to_fraction_maybe(x, **param)
                    if success: # job is done
                        latex = fraction_to_latex(frac, **param)
                        approx = frac
                    else: # not a fraction
                        # we will explore more complicated structures
                        # maybe it is a product of fraction and square root?
                        if param['numbertype'] in ['root', 'algebraic']: # do we look for complicated stuff?
                            # we first look for a fraction times a root (that's the 'root' parameter)
                            root, success, latex = real_to_fraction_x_root_maybe(x, **param)
                            if success:
                                # we already have latex
                                approx = root
                            else:
                                # maybe it is a more convoluted rational combination of different roots?
                                if param['numbertype'] == 'algebraic': # do we want to find such thing?
                                    algebraic, success, latex = real_to_simple_algebraic_maybe(x, **param)
                                    if success:
                                        # we already have latex
                                        approx = algebraic
                        if not success: 
                            # well there is nothing we can do except rounding it now 
                            # we have failed to simply approximate our number x 
                            # we convert it to a Decimal and raise a warning  
                            latex = param['frmt'].format(x) # shorten useless zeros
                            approx = float(latex) # same
                            warning_text = f"The real number {x} couldn't properly be converted into a simple algebraic expression up to the specified tolerance level tol={param['tol']}. It is left approximated in decimal form : {approx}."
                            warn(warning_text)
                            
            else: # THIRD : we have a Complex number to deal with
                # we simply divide the work in two : real and imaginary parts
                latex_real, approx_real = number_to_algebraic(x.real, **param)
                latex_imag, approx_imag = number_to_algebraic(x.imag, **param)
                latex = latex_real + ' + ' + latex_imag + ' i'
                approx = complex(approx_real, approx_imag)
        else: # a Number which is not a Complex? Idk what it is
            raise ValueError('You gave me a number which is not a Complex')
    else: # not a Number (arrays, lists and stuff should be treated beforehand, see "latexifier")
        raise ValueError('You gave me something which is not a number')
    
    return latex, approx
    

        
def number_to_latex(x, **param):
    return number_to_algebraic(x, **param)[0]

def simplify_number(x, **param):
    return number_to_algebraic(x, **param)[1]


#-------------------------------------------------------
# Functions to deal with structured data
#-------------------------------------------------------    

def numpyarray_to_latex(a, **param):
    # taken from https://github.com/josephcslater/array_to_latex
    if len(a.shape) > 2:
        raise ValueError('You gave me an array which has more than two dimensions : I cannot turn it into a latex array')
    if len(a.shape) == 1:
        a = np.array([a]) # make it a 2D row matrix
        if param['column']:
            a = a.T # a 2D column matrix
    # from now we have a 2D array    
    
    if param['newline']:
        end = '\n'
    else:
        end = ''
        
    approx = np.zeros_like(a) # begins the numpy array
    latex = r'\begin{' + param['arraytype'] + '} ' + end # begins the latex array
    for i in np.arange(a.shape[0]):
        for j in np.arange(a.shape[1]):
            latex_ij, approx_ij = latexifier(a[i,j], **param)
            latex += latex_ij + r' & '
            approx[i,j] = approx_ij
        latex = latex[:-2] # delete the last ' & '
        latex = latex + '\\\\ ' + end # ends the line
    latex = latex[:-3]  # delete the last ' \\ '
    latex = latex + end + r'\end{' + param['arraytype'] + '}' # ends the latex array
    
    return latex, approx


def list_to_latex(L, **param):
    """ returns the elements of the list one next to each other """
    approx = np.zeros(len(L)) # begins a numpy array
    latex = ''
    for i in range(len(L)):
        latex_i, approx_i = latexifier(L[i], **param)
        latex += latex_i + param['list_separator']
        approx[i] = approx_i
    latex = latex[:-len(param['list_separator'])]  
    approx = list(approx)
    return latex, approx

def tuple_to_latex(tup, **param):
    """ returns the elements of the tuple one next to each other, in the form (..,..,..) """
    approx = np.zeros(len(tup)) # begins a numpy array
    latex = '\\left('
    for i in range(len(tup)):
        latex_i, approx_i = latexifier(tup[i], **param)
        latex += latex_i +', '
        approx[i] = approx_i
    latex = latex[:-2] + ' \\right)'
    approx = tuple(approx)
    return latex, approx

def sympy_to_latex(poly, **param):
    """ Here we try to handle sympy formal expressions. We certainly miss a lot of cases.
        So far we treat well the polynomials ; if not polynomial (like a number) we pray for sympy to do a good job
        
        In sympy there is a bunch of notions of 'polynomials'
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
        dico = poly.as_dict() # a dict is easy to parse and loop over
        approx_dico = { key : simplify_number(dico[key], **param) for key in dico.keys() } # simplify coefficients
        approx = sympy.Poly.from_dict(approx_dico, poly.gens)
        latex = sympy.latex(approx.as_expr())
    else: 
        try: # if it is an expression we can try to convert it to a polynomial
            proper_poly = sympy.Poly(poly)
        except: # if the conversion failed
            latex = sympy.latex(poly) # we just let sympy do something
            approx = poly # we keep whatever input we had
            warn(f" latexify received the object {poly} but I do not recognize it. I latexify it as :\n{latex}")
        else: # if the conversion succeeded we latexify the polynomial
            latex, approx = sympy_to_latex(proper_poly, **param)
    latex = latex.replace(' ','') # superfluous spaces
    return latex, approx



#-------------------------------------------------------
# Main functions
#-------------------------------------------------------    
    
def latexifier(x, **param):
    approx = None
    if isinstance(x, list):
        latex, approx = list_to_latex(x, **param)
    elif isinstance(x, tuple):
        latex, approx = tuple_to_latex(x, **param)
    elif isinstance(x, np.ndarray):
        latex, approx = numpyarray_to_latex(x, **param)
    elif isinstance(x, numbers.Number):
        latex, approx = number_to_algebraic(x, **param)
    elif isinstance(x, sympy.Basic):
        latex, approx = sympy_to_latex(x, **param)
    else:
        raise ValueError('Unknown data type. Can be: numbers, arrays, lists, or basic sympy objects')
    return latex, approx

def latexify(x, **param):
    param = get_parameters(**param)
    latex, approx = latexifier(x, **param)
    
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
        
    if param['verbose'] == True:
        print(latex)
    elif param['verbose'] in {'printmk', 'markdown'}:
        if param['mathmode'] == 'raw':
            printmk('$'+latex+'$')
        else:
            printmk(latex)
    
    if param['value'] == True:
        return latex, approx
    else: # default
        return latex