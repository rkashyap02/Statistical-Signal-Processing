# Python function to implement Woodbury's Matrix Inversion Lemma
import numpy as np
from numpy.linalg import inv
def woodbury_inversion(A,U,C,V):
    """
    ===========================
    Woodbury Rank-k Inverse fix
    ===========================
    NAME:
    woodbury_inversion
    
    SYNOPSIS:
    A_inv = woodbury_inversion(A, U, C, V)
    
    DESCRIPTION:
    Program to find inverse after a rank-k correction
    Refer - https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    
    INPUTS:
    A           : Input n x n array
    U           : Input k x n array
    C           : Input k x k array
    V           : Input n x k array
    
    OUTPUTS:
    A-inv       : Inverse of the matrix A after a rank-k fix
    
    AUTHOR:
    Rohit Kashyap, 2017
    """
    
    # Check dimensionality of A, U, C, V
    if(A.shape[0] != A.shape[1]):
        return "Check the input Matrix A"
    # Matrix C is k x k
    if(C.shape[0] != C.shape[1]):
        return "Check the input Matrix C"
    # Matrix U is n x k and V is k x n
    if(U.shape[0] != V.shape[1] or U.shape[1] != V.shape[0]):
        return "Check Matrix U or V"
    # Check dimensionality with C
    if(U.shape[1] != C.shape[0] or V.shape[0] != C.shape[0]):
        return "Check Matrix U or V"
    # Check if A,U,C,V are numpy matrices
    A, U, C, V = np.matrix(A), np.matrix(U), np.matrix(C), np.matrix(V)
    # Check for a simpler rank-k correction involving U, C and V
    if(U.shape[0] == 1 and V.shape == 1 and C == 1):
        # Transpose row vectors for simplicity
        U, V = U.T, V.T
        # Return the simplified matrix
        return (np.identity(A.shape[0])-(1/(1+V.T*U))*(U*V.T))
    else:
        # Return the inverse rank-k correction of matrix A
        return inv(A)-inv(A)*U*inv(inv(C)+V*inv(A)*U)*V*inv(A)
