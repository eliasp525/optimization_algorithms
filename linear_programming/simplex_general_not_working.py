import numpy as np

# LINEAR PROGRAMMING

# BFP - basic feasible point
# - x feasible
# i not in B(x) => x_i = 0
# B = [A_i]_{i in B(x)} nonsingular, B in R^{m*m}



def simplex_linear(c_in : np.matrix, Aiq : np.matrix = None, biq : np.matrix = None, Aeq : np.matrix = None, beq : np.matrix = np.matrix([])):
    '''
    Object function: c_in^T x,
    Aiq*x <= biq,
    Aeq*x  = beq,
    x     >= 0
    --------------------------
    Theorem 13.4: If an LP is bounded and non-degenerate, the Simplex method terminates
    at a BOP
    --------------------------
    Simplex method iterates BFPs until one that fulfills KKT(1-5) is found.
    Each step is a move from a vertex to a neighboring vertex (one change in the basis), that
    decreases the objective.
    '''

    # Standardizing the problem
    # x = [x+ x- z]^T
    # Object function: [c_in^T -c_in^T 0] x
    # A x  = b,
    # x   <= 0
    inequalities = np.size(Aiq) > 0 and np.size(biq) > 0
    equailites = np.size(Aeq) > 0 and np.size(beq) > 0
    
    if inequalities and equailites:
        A = np.block([
            [Aiq,     -Aiq, np.eye(np.size(Aiq, 0))                       ],
            [Aeq, -Aeq, np.zeros((np.size(Aeq, 0), np.size(Aeq, 1)))]
        ])
        b = np.block([
            [biq],
            [beq]
        ])
    elif inequalities:
        A = np.block([Aiq, -Aiq, np.eye(np.size(Aiq, 0))])
        b = biq
    elif equailites:
        A = np.block([Aeq, -Aeq])
        b = beq

    print("A:\n", A)
    print("b:\n", b)

    nx = np.size(A, 1)
    print(nx)
    # c = [c_in^T -c_in^T 0]
    c_in_T = np.transpose(c_in)
    num_slack_vars = nx-2*np.size(c_in_T, 1)
    c = np.transpose(np.block([c_in_T, -c_in_T, np.zeros((1, num_slack_vars))]))
    print("c:\n", c)

    # Intializing sets
    indices = np.linspace(0, nx-1, nx, dtype=int)
    print("indices:", indices)
    x = np.array(np.zeros((np.size(c), 1)))
    basis_set = [4, 5] # example x = [5, 8, 0, 0] and basis_set = [3, 4]
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

    print(np.linalg.inv(np.transpose(B)))
    print(c_B)

    iterations = 0
    while True:
        iterations += 1
        print(f"============ Iteration {iterations} ==============")
        x[basis_set] = np.linalg.inv(B)*b
        print("x[basis_set]:\n", x[basis_set])
        lmda = np.linalg.inv(np.transpose(B))*c_B
        print("lambda:\n", lmda)
        s_N = c_N - np.transpose(N)*lmda
        print("s_N:\n", s_N)
        
        if np.all(np.array(s_N) >= 0):
            print("Found solution!")
            print(x)
            break
        
        q = active_set[np.argmin(s_N)] # q = index to move from N to B.
        print(f"Found q:{q}, from value {s_N[np.argmin(s_N)]}")
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

        # update variables
        B = A[:, basis_set]
        N = A[:, active_set]
        c_B = c[basis_set, :]
        c_N = c[active_set, :]

        print("basis_set:", basis_set)
        print("active_set:", active_set)

    # extract solution
    n_c_in = np.size(c_in, 0)
    x_pos = x[:n_c_in]
    x_neg = x[n_c_in:2*n_c_in]
    # slacks = x[2*n_c_in:]
    return x_pos-x_neg


# def find_initial_bfp(c: np.array, Aiq : np.matrix, biq : np.matrix, Aeq : np.matrix, beq : np.matrix):
    # return [3, 4] # not proper way to do this

solution = simplex_linear(np.transpose(np.matrix([-4, -2])), np.matrix([[1, 2],[2, 0.5]]), np.transpose(np.matrix([5, 8])))
print("Solution: ", solution)

# active set method
def linear_active_set(c : np.array, A : np.matrix, b : np.matrix, Aeq : np.matrix, beq : np.matrix):
    pass