import numpy as np

from c_gammas import c_gammas  # 计算迭代过程中需要的参数


def CF_ADI(A, B, k, epsilon1, epsilon2):
    """
    求解Lyapunov方程：A'X+XA+BB'=0 的一个近似解的Cholesky因子Z
    Parameters：
    A：n*n阶实的稳定矩阵
    B：n*r阶的列满秩矩阵
    k：进行ADI迭代的最大允许迭代次数
    epsilon1,epsilon2：两个精度要求
    Return：
    Z：方程的近似解的Cholesky因子，满足X=ZZ'
    """

    # 计算k个参数
    gamma = c_gammas(A, k)

    I = np.eye(A.shape[0])
    Y = (-2*gamma[0])**(1/2)*np.linalg.inv(A+gamma[0]*I)*B
    Z = Y
    for k in range(1, k):
        Y = (-2*gamma[k])**(1/2)*np.linalg.inv(A+gamma[k]*I)*B
        T = np.linalg.inv(A+gamma[k]*I)*(A-gamma[k]*I)*Z
        y = np.linalg.norm(Y, 2)  # 矩阵的二范数
        z = np.linalg.norm(Y, 2)
        Z = np.hstack([Y, T])  # 水平拼接
        if (y <= epsilon1) and (y/z <= epsilon2):
            return(Z)
    return(Z)
