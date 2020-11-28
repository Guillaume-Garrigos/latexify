import numpy as np
import numbers
import fractions
from fractions import Fraction
from latexify.CFraction import CFraction

def printmk(*tuple_of_text):
    from IPython.display import display, Markdown
    L = [Markdown(text) for text in tuple_of_text]
    return display(*tuple(L))

def default_parameters():
    return {
            'denominator_max' : 10, # maximal denominator allowed in fractions
            'radical_max' : 7, # We allow for sqrt(n) to show up, with n <= radical_max
            'style_fraction' : 'frac', # how to display fractions 
            'frmt' : '{:3.6f}', # format for rounding reals when no nice approximation is found
            'newline' : False, # should the latex string contain \n newline characters?
            'arraytype' : 'pmatrix', # arrays can be converted in many latex flavours
            'list_separator' : ', '
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
            return (frac, n), True
    return (x, 1), False

def real_to_simple_radical_maybe(x, **param):
    # tries to convert a real to something like a+b*sqrt(n)
    # where a and b are Fractions with bounded denominator
    # 'n' cannot be larger than 'radical_max'
    # Here is a stupid implementation, surely we can do better
    param = get_parameters(**param)
    frac, success = real_to_fraction_maybe(x, **param)
    if success:
        return (frac, 0, 1), True
    # now we know that x = a+b*sqrt(n) with b != 0, n!=0
    # the numerator for a is in the worst case x*denominator_max
    # the numerator for b is in the worst case x*denominator_max/sqrt(n)
    numerator_max = int(np.ceil(x * param['denominator_max']))
    for i in range(-numerator_max, numerator_max+1):
        for j in range(1, param['denominator_max']+1):
            radical, success = real_to_fraction_x_root_maybe(x - Fraction(i,j), **param)
            if success:
                return (Fraction(i,j), radical[0], radical[1]), True
    return (x, 1, 1), False



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
        
# this function share a lot (of issues) with a function displaying a polynomial
# in particular with all the story of dealing with negative coefficients etc

def simple_radical_to_latex(x, **param):
    """ style_fraction can be 'frac', 'dfrac' or 'inline' 
        x is a triplet (a,b,c) with x = a+b*sqrt(c)
        a and b are Fraction, c is integer
    """
    param = get_parameters(**param)
    a = x[0]
    b = x[1]
    c = x[2]
    latexa = fraction_to_latex(a, **param)
    latexb = fraction_to_latex(b, **param)
    latexc = '\\sqrt{' + str(c) + '}'
    # we want to treat nicely the cases a=0, b=1, b<0, c=0, c=1
    if c*b == 0: # x = a
        return latexa
    if c == 1: # x = a+b
        return fraction_to_latex(a+b, **param)
    if b == 1: 
        if a == 0: # x = sqrt(c)
            return latexc
        else: # x = a + sqrt(c)
            return latexa + '+' + latexc
    elif b > 0: 
        if a == 0: # x = b*sqrt(c)
            return latexb + latexc
        else: # x = a + b*sqrt(c)
            return latexa + '+' + latexb + latexc
    else: # b<0 we need a -sign
        latexbminus = fraction_to_latex(-b, **param)
        if a == 0: # x = b*sqrt(c)
            return latexb + latexc
        else: # x = a + b*sqrt(c)
            return latexa + '-' + latexbminus + latexc
        
        
def number_to_latex(x, verbose=False, **param):
    param = get_parameters(**param)
    approx = x
    latex = ''
    if isinstance(x, numbers.Number):
        if isinstance(x, numbers.Complex):
            if isinstance(x, numbers.Real):
                if isinstance(x, numbers.Rational):
                    if isinstance(x, numbers.Integral): # x is an integer TRICKY : https://stackoverflow.com/questions/48458438/why-is-numpy-int32-not-recognized-as-an-int-type
                        latex = str(x)
                    elif isinstance(x, Fraction): # x is a Fraction
                        latex = fraction_to_latex(x, **param)
                    else:
                        raise ValueError('You gave me a Rational which is neither integer or Fraction')
                else: # x is complicated to deal with
                    # maybe it is a simple fraction in disguise
                    frac, success = real_to_fraction_maybe(x, **param)
                    if success:
                        latex = fraction_to_latex(frac, **param)
                        approx = frac
                    else:
                        # maybe it is a combination of fraction and square root
                        algebraic, success = real_to_simple_radical_maybe(x, **param)
                        if success:
                            latex = simple_radical_to_latex(algebraic, **param)    
                        else: # well we can just round it now              
                            latex = param['frmt'].format(x) # need to shorten useless zeros
                            approx = float(latex)
                            warn("A float got rounded")
            else: # we have a complex number to deal with
                latex_real, approx_real = number_to_latex(x.real, verbose=True, **param)
                latex_imag, approx_imag = number_to_latex(x.imag, verbose=True, **param)
                latex = latex_real + ' + ' + latex_imag + ' i'
                approx = CFraction(approx_real, approx_imag)
        else:
            raise ValueError('You gave me a number which is not a Complex')
    else:
        raise ValueError('You gave me something which is not a number')
    if verbose:
        return latex, approx
    else:
        return latex
    

    
def latexify(x, verbose=False, **param):
    param = get_parameters(**param)
    if isinstance(x, list):
        latex = list_to_latex(x, **param)
    elif isinstance(x, tuple):
        latex = tuple_to_latex(x, **param)
    elif isinstance(x, np.ndarray):
        latex = numpyarray_to_latex(x, column=True, **param)
    elif isinstance(x, numbers.Number):
        latex = number_to_latex(x, verbose=False, **param)
    else:
        raise ValueError('Unknown data type. Can be: numbers, arrays, lists')
        
    if verbose == True:
        print(latex)
    elif verbose in {'printmk', 'markdown'}:
        printmk('$'+latex+'$')
        
    return latex

# taken from https://github.com/josephcslater/array_to_latex

def numpyarray_to_latex(a, column=True, **param):
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
            out = out + latexify(a[i,j], **param) + r' & '
        out = out[:-2]
        out = out + '\\\\ ' + end
    out = out[:-5] + end + r'\end{' + param['arraytype'] + '}'
    return out


def list_to_latex(L, **param):
    """ returns the elements of the list one next to each other """
    param = get_parameters(**param)
    latex = ''
    for x in L:
        latex = latex + latexify(x, **param) + param['list_separator']
    return latex[:-len(param['list_separator'])]  

def tuple_to_latex(tup, **param):
    """ returns the elements of the tuple one next to each other, in the form (..,..,..) """
    param = get_parameters(**param)
    latex = '\\left('
    for x in tup:
        latex = latex + latexify(x, **param)+', '
    latex = latex[:-2] + ' \\right)'
    return latex    