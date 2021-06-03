import numpy as np
from scipy.linalg import schur

from vec import vec  # 通过拉直求解方程 AX+XB=C


def bs_lyap(A, Q):
    """
    利用Bartels-Stewart算法解Lyapunov方程 A'X+XA+Q=0，但针对Lyapunov方程
    的特殊性进行了一定的优化

    Parameters
    ----------
    A : (n,n) 实矩阵，并且其任意两个特征值之和不为0
    Q : (n,n) 实对称矩阵

    Returns
    -------
    output : 实对称解((n,n)ndarry)
    """

    n = A.shape[0]
    # 计算 A 的 Schur 分解 A = UTU'
    T, U = schur(A)

    # 确定T每个对角块的阶数
    N = [0]  # N[i] 为 T 第 i 个对角块的阶数
    i = 0   # 数据初始化
    while i < n-1:
        if T[i+1, i] == 0:  # 该对角块阶数为1
            N.append(1)
            i += 1
        else:
            N.append(2)
            i += 2
    if i == n-1:
        N.append(1)
    m=np.cumsum(N)  # M[i] 为 T 前 i 个对角块的阶数和

    # 将A=U'TU带入方程中得：T'Y+YT=-W
    # 其中：Y=U'XU,W=U'QU
    W = (U.T).dot(Q.dot(U))
    Y = np.zeros((n, n))
    p = len(N)-1
    for j in range(p):
        for i in range(j, p):
            F = -W[m[i]:m[i+1], m[j]:m[j+1]] - \
                Y[m[i]:m[i+1], :m[j]].dot(T[:m[j], m[j]:m[j+1]]) - \
                (Y[m[j]:m[j+1], :m[i]].dot(T[:m[i], m[i]:m[i+1]])).T
            Y[m[i]:m[i+1], m[j]:m[j+1]] = vec(T[m[i]:m[i+1], \
                     m[i]:m[i+1]].T, T[m[j]:m[j+1], m[j]:m[j+1]], F)
    Y = np.tril(Y)+np.tril(Y, -1).T
    return U.dot(Y.dot(U.T))
