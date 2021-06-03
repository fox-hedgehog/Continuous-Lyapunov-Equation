import numpy as np


def conjugate_gradient(A, Q, epsilon, km):
    """
    通过共轭梯度法求解Lyapunov方程:A'X+XA+Q=0

    Parameters
    ----------
    A : 正定矩阵
    Q : 对称矩阵
    epsilon : 精度要求 s.t.||A'X+XA+Q||_F/||Q||_F<=epsilon
    km : 最小允许迭代次数
    Returns
    -------
    X : 满足精度要求的近似解
    """
    
    X = np.zeros_like(A)  # 给定初值零矩阵
    R = -Q
    rho = np.trace(Q.dot(Q))
    sigma = rho**(1/2)
    k = 0
    rho_ = 0  # 防止pytho提示未定义rho_
    while (rho**(1/2) > sigma*epsilon)&(k < km):
        k += 1
        if k == 1:  # 共轭梯度方向p^(k-1)
            P = R
        else:
            beta = rho/rho_  # beta^(k-1)
            P = R+beta*P
        W = A.dot(P)+P.dot(A)
        alpha = rho/np.trace(P.dot(W))  # alpha^(k-1)
        X = X+alpha*P  # X^k
        R = R-alpha*W  # R^k
        rho_ = rho.copy()  # <R^(k-1),R^(k-1)>
        rho = np.trace(R.dot(R))  # <R^k,R^k>
    return X
