## Unconstrained optimization problem ##
'''
min { f(x) }
 x
(no constraints)
'''

## General linesearch algorithm ##
'''
1. Inital guess
2. while termination criteria not fulfilled:
    a) Find DESCENT DIRECTION pk from xk
        * Steepest direction
        * Newton
        * Quasi-Newton
    b) Find appropriate step legnth ak and set x_{k+1} = xk + ak * pk
    c) k = k + 1
3. Check conditions for optmiality.

Possible termination criteria:
* Max iteration
* stopped making progress (below epsilon) either in x or f(x)
* necessary condition: gradient(f(x)) < epsilon
'''

## Conditions for optimiality ##
'''
Necessary: x* local solution -> gradient(f(x*)) = 0

Sufficient: gradient(f(x*)) = 0 and hessian(f(x*)) > 0 -> strict local solution
'''

## Scaling ##
'''
In some cases the opbject function changes much faster in some directions. 
We say that the problem is poorly scaled.
Poor scaling will affect Steepest descent, but not Newton.
'''

## Step length ##
'''
How do we choose ak?

Following the wolfe conditions!
1. Sufficient decrease (Armijo condition)
2. Desired slope (Curevature condition)

Finding such as step can be done by backtracking line search.
(Algorithm 3.1 in Nocedal)

'''

def newton_line_search():
    '''
    Newton line search with Hessian Modification
    ============================================
    Algorithm 3.2 in Nocedal
    ============================================
    TODO: implement!
    '''
    pass


## Quasi Newton ##
'''
A family of methods where hessian is approximated with 
only gradient information to reduce computational complexity
=============================================================
'''

def quasi_newton_BFGS():
    '''
    Broyden Fletcher Goldfarb Shanno
    ================================
    Considered the most effective Q-N formula!
    (Limited memory BFGS better, especially when H becomes big)
    Algorithm 6.1 in Nocedal
    Update equation: Eq. 6.17 in Nocedal
    Implementation notes on page 142.
    ================================
    TODO: implement!
    '''
    pass

def quasi_newton_DFP():
    '''
    David Fletcher Powell
    ======================
    Use inverse update formula for efficiency.
    Equation 6.15 in Nocedal
    ======================
    TODO: implement!
    '''
    pass

