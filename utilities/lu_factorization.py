import numpy as np
# Solving linear systems is an expensive operation used in optimization algorithms
# E.g. it happens two times each iteration of simplex method


'''
Given the system
B x = b
we factorize
L U = B (Eq. 13.26)
where L is lower triangular and U is upper triangular
======================================================
The system can then be solved as two equations:
L xbar = b
U x = xbar
======================================================
These can be updated, saving computational complexity (time) 
compared to solving the linear system from scratch each time.
======================================================
'''



'''
Updating in the simplex method
==============================
From 13.26:
U = inv(L) B
==============================
B is updated by replacing column p with A_q.
Call the updated matrix B+.
inv(L) B+ then becomes upper triangular except for column p,
which is replaced with inv(L) A_q.
==============================
We now perform a cyclic permutation that moves column p to the last column position
m and moves columns p + 1, p + 2, . . . , m one position to the left to make room for it.
This will make room for the new column, but will still not restore the upper triangular matrix U.
==============================
'''
def cyclic_permutation_matrix(size, p):
    '''
    Returns permutation matrix that moves column p to the last column 
    position m(=size) and moves columns p + 1, p + 2, . . . , m one position to the left
    p = 0 => Col 1
    '''
    P = np.eye(size, size, dtype=int)
    one_step_left = np.matrix(P[:, p+1:])
    P[:, size-1] = P[:, p] # moving column p all the way right
    P[:, p:(size-1)] = one_step_left
    return P

print("Cyclic permutation matrix P:")
P = cyclic_permutation_matrix(5, 1)
print(P)
'''
A permutation matrix P can be used to achieve this.
This is just matrix that reorders rows (right multiply) 
or columns (left multiply) when applied to another.
'''
# P = np.matrix([
# [1, 0, 0, 0, 0], 
# [0, 0, 0, 1, 0], 
# [0, 1, 0, 0, 0], 
# [0, 0, 0, 0, 1], 
# [0, 0, 1, 0, 0]
# ])
# print("Permutation matrix:\n", P)

K = np.matrix([
[1,   2,  3,  4,  5], 
[6,   7,  8,  9, 10], 
[11, 12, 13, 14, 15], 
[16, 17, 18, 19, 20], 
[21, 22, 23, 24, 25]
])
print("Matrix K:\n", K)
print("Switching columns, K * P:\n", K*P)
print("Switching rows, P * K:\n", P*K)
'''
In the example in Nocedal column p 
is moved to the far right (P1 is applied),
then row p is moved to the bottom (P1 transposed is applied).
==============================
Restore upper triangular form by sparse Gaussian elimination on P1 inv(L) B+ P1^T. 
I.e. we ﬁnd L1 and U1 (lower and upper triangular, respectively)
such that
P1 inv(L) B+ P1^T = L1 U1
==============================
'''



'''
Below example in Nocedal (p. 374-375)
The procedure we have just outlined is due to Forrest and Tomlin [110]. It is quite
efﬁcient, because it requires the storage of little data at each update and does not require much
movement of data in memory. Its major disadvantage is possible numerical instability. Large
elements in the factors of a matrix are a sure indicator of instability, and the multipliers in
the L 1 factor (l 52 in (13.30), for example) may be very large.
'''