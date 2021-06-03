import numpy as np
from scipy.linalg import schur, qr, solve, cholesky

from vec import vec


def hammarling(A, C):
    """
    利用Hammarling算法求解Lyapunov方程：AX+XA'+C'C=0，

    Parameters
    ----------
    A：n*n阶实的稳定矩阵
    C：l*n阶实矩阵，并且满足(C,A)可观测

    Returns
    -------
    output : 对角元均为正数的上n阶上三角矩阵
        Lyapunov方程的解 X 的 Cholesky 分解
    """

    n = A.shape[0]
    l = C.shape[0]  # 矩阵 C 的行数
    # 对 A 进行实 Schur 分解， A = UTU'
    T, U = schur(A)
    # 对 CU 进行 QR 分解，R : l*n上三角矩阵
    R = qr(C.dot(U), mode='r')[0]

    # 确定T每个对角块的阶数
    N = []  # N[i]即为T第 i+1 个对角块的阶数
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

    Y = np.zeros(A.shape)
    k = 0  # 记录Y已经求完的行数
    m = 0  # 记录正在求 Y 的第几个块行
    for nj in N:  # nj 为第 j+1 个块行的阶数
        m += 1
        Tjj = T[k:k+nj, k:k+nj]
        Tj2 = T[k:k+nj, k+nj:]
        T22 = T[k+nj:, k+nj:]
        r = min(l, nj)
        Rjj = R[:r, :nj]
        Rj2 = R[:r, nj:]
        # 通过拉直求解 Tjj'Z + ZTjj = -Rjj'Rjj，其中 Z = Yjj'Yjj
        Yjj = cholesky(vec(Tjj.T, Tjj, -Rjj.T.dot(Rjj)))

        Yjj_ = np.linalg.inv(Yjj)  # Yjj的逆
        Omega = Rjj.dot(Yjj_)
        # 通过分块求解(YjjTjjYjj_)'Yj2+Yj2T22=-(YjjTj2+Omega'Rj2) 得 Yj2
        Yj2 = np.empty_like(Tj2, float)
        D = Yjj.dot(Tjj.dot(Yjj_)).T
        E = -(Yjj.dot(Tj2)+Omega.T.dot(Rj2))
        q = 0
        for ni in N[m:]:  # 通过拉直求解方程
            I = np.eye(ni)
            B = np.kron(I, D)+np.kron(T22[q:q+ni, q:q+ni].T, I)
            F = E[:, q:q+ni]-Yj2[:, :q].dot(T22[:q, q:q+ni])
            Yj2[:, q:q+ni] = solve(B, F.ravel('F')
                                   ).reshape((ni, -1), order='F')
            q += ni

        # 将求得的值赋给 Y
        Y[k:k+nj, k:k+nj] = Yjj
        Y[k:k+nj, k+nj:] = Yj2
        # 更新方程
        V = Rj2-Omega.dot(Yj2)
        if r == l:
            W = V
        else:
            W = np.vstack([V, R[nj:, nj:]])
        R = qr(W)[1]

        k = k+nj
    # 对 YU' 进行 QR 分解
    R1 = qr(Y.dot(U.T), mode='r')[0]  # R1 : n 阶上三角矩阵
    # 将 G 对角线元素为负的行变号
    H = np.diag(np.array(np.diagonal(R1) < 0))
    return (np.eye(n)-2*H).dot(R1)
