import numpy as np


def vec(A, B, C):
    """
    通过Kronecker积和拉直求解Sylvester方程:AX+XB=C

    Parameters
    ----------
    A : m阶矩阵
    B : n阶矩阵
    C : (m,n)阶矩阵

    Returns
    -------
    output : (m,n)阶矩阵
        方程的解
    """

    # 拉直化为 By=b
    I1 = np.eye(A.shape[0])
    I2 = np.eye(B.shape[0])
    B = np.kron(I1, A)+np.kron(B.T, I2)
    b = C.ravel('F')  # 按列拉直
    y = np.linalg.solve(B, b)
    return y.reshape((C.shape), order='F')
