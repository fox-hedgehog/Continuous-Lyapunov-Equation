# 问题记录？？？
# 判断AX运算量
# 要对三对角化产生的矩阵R求逆，这对于高阶矩阵工作量较大，可修改三对角矩阵程序
# 该算法中两次涉及了求解线性方程组，这对于大方程的工作量也是特别大的
import numpy as np

from tridiagonalization import tridiagonalization as tri  # 矩阵三对角化
from c_gammas import c_gammas  # 计算迭代过程中需要的参数


def ADI(A, Q, k,  M1=1e-3, M2=1e-3):
    """
    使用交替方向法求解Lyapunov方程 A'X+XA+Q=0

    Parameters
    ----------
    A : n*n阶实的稳定矩阵
    Q : n*n阶实对称矩阵
    k : 迭代次数
    M1 : float,optional
        在需要三对角化时，一个乘子最大允许的上界
    如下为计算系数过程中可能用到的参数  
    M2 : float,optional
        删除列表中近似元素时所允许的误差范围

    Returns
    -------
    output : 迭代k次后的近似解
    """

    n = A.shape[0]
    I = np.eye(n)
    R = I.copy()
    X = np.zeros(A.shape)  # 初始解

    # 判断 A 是否为稀疏矩阵(A的非零元占比小于10%)
    p = np.sum(A != 0)/n**2
    if p <= 0.1:
        A, R = tri(A, M1)  # 对A进行三对角化
        R_ = np.linalg.inv(R)
        # Construct Q = R'QR
        Q = R_.T.dot(Q.dot(R_))
    
    # 计算k个参数
    gamma = c_gammas(A, k, M2)
    for k in range(k):
        # 求解X_(j-1/2)
        X = np.linalg.solve(A.T+gamma[k]*I, -Q-X.dot(A-gamma[k]*I))
        # 求解X_j
        X = np.linalg.solve(A.T+gamma[k]*I, -Q-X.T.dot(A-gamma[k]*I))
    if (R == I).all():
        return X
    return R.T.dot(X.dot(R))
