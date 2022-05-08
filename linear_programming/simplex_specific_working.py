import numpy as np

# LINEAR PROGRAMMING

# BFP - basic feasible point
# - x feasible
# i not in B(x) => x_i = 0
# B = [A_i]_{i in B(x)} nonsingular, B in R^{m*m}



def simplex_linear():
    '''
    Active set methods (such as simplex method) maintains explicitly an estimate of 
    the set of inequality constraints that are active at the solution (the set fancyN
    for the Simplex method)
    --------------------------
    Theorem 13.4: If an LP is bounded and non-degenerate, the Simplex method terminates
    at a BOP
    --------------------------
    Simplex method iterates BFPs until one that fulfills KKT(1-5) is found.
    Each step is a move from a vertex to a neighboring vertex (one change in the basis), that
    decreases the objective.
    --------------------------
    Typically, at most 2m to 3m iterations (where m is dimension of x ???)
    Worst case: All vertices must be visited (exponential complexity in n)
    '''

    A = np.matrix([[1,   1, 1, 0], 
                   [2, 0.5, 0, 1]])
    
    b = np.transpose(np.matrix([5, 8]))

    print("A:\n", A)
    print("b:\n", b)

    c = np.transpose(np.matrix([-4, -2, 0, 0]))

    nx = np.size(A, 1)
    print(nx)
    
    # Intializing sets
    indices = np.linspace(0, nx-1, nx, dtype=int)
    print("indices:", indices)
    x = np.array(np.zeros((np.size(c), 1)))
    basis_set = [0, 1] # example x = [5, 8, 0, 0] and basis_set = [3, 4]
    active_set = [i for i in indices if i not in basis_set]
    print("basis_set:", basis_set)
    print("active_set:", active_set)    
    # partition A
    B = A[:, basis_set] # B(x)
    N = A[:, active_set] # N(x)
    print("B:\n", B)
    print("N:\n", N)
    # partition c
    c_B = c[basis_set, :]
    c_N = c[active_set, :]
    print(f"c_N:\n {c_N}")

    iterations = 0
    while True:
        iterations += 1
        print()
        print(f"================ Iteration {iterations} ===================")
        x[basis_set] = np.linalg.inv(B)*b
        print("x[basis_set]:\n", x[basis_set])
        print(f"B before:\n{B}")
        lmda = np.linalg.inv(np.transpose(B))*c_B
        print("lambda:\n", lmda)
        s_N = c_N - np.transpose(N)*lmda
        print("s_N:\n", s_N)
        
        if np.all(np.array(s_N) >= 0):
            print("Found solution!")
            break
        
        q = active_set[np.argmin(s_N)] # q = index to move from N to B.
        print(f"Lowest value of s_N is {s_N[np.argmin(s_N)]}, found at index q: {q}")
        # p = index to move from B to N.
        # increase x[q] while Ax = b and move the i of x[i] that becomes zero first from B to N.
        
        d = np.linalg.inv(B)*A[:, q]
        print("d:\n", d)
        if np.all(d <= 0):
            print("Problem is unbounded")
            break
        
        steplengths = [x[basis_set][i, :]/d_i for i, d_i in enumerate(np.array(d).flatten())]
        x[q] = np.min(steplengths)
        print(f"new x[q]: {x[q]}")
        p = basis_set[np.argmin(steplengths)]
        print(f"Found p: {p}")

        x[basis_set] = x[basis_set] - d*x[q]
        x[active_set] = np.transpose(np.block([np.zeros((1, q)), x[q], np.zeros((1, np.size(N, 1)-1-q))]))
        print(f"new x[active_set]:\n{x[active_set]}")

        # updating sets
        basis_set.append(q)
        basis_set.remove(p)
        basis_set.sort()
        active_set.remove(q)
        active_set.append(p)
        active_set.sort()

        B = A[:, basis_set]
        N = A[:, active_set]
        c_B = c[basis_set, :]
        c_N = c[active_set, :]

        print("basis_set:", basis_set)
        print(f"B: {B}")
        print("active_set:", active_set)
        print(f"N: {N}")

    return x

solution = simplex_linear()
print(f"Solution {solution}")