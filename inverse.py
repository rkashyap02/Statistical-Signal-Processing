# Function to compute inverse of any matrix
import numpy as np
from numpy.linalg import inv

def inverse(A):
    """
    ======================
    Inverse of any matrix
    ======================
    NAME:
    inverse
    
    SYNOPSIS:
    A_inv = inverse(A)
    
    DESCRIPTION:
    Program to find inverse of any matrix
    
    INPUTS:
    A           : Input matrix
    
    OUTPUTS:
    A-inv       : Inverse or the pseudo inverse of the matrix A 
    
    AUTHOR:
    Rohit Kashyap, 2017
    """
    
    # Convert numpy array to matrix
    A = np.matrix(A)
    # Check for a square matrix
    if(A.shape[0]==A.shape[1]):
        if(np.linalg.det(A) == 0):
            return "Singular Matrix"
        else:
            return inv(A)
    # Check for an Undetermined system
    elif(A.shape[0] < A.shape[1]):
        # Find the Pseudo inverse from A using the minimum norm 
        return A.T*inv(A*A.T)
    # This is for an Overdetermined system
    else:
        # This is the least square solution to an Overdetermined system
        return inv(A.T*A)*A.T
