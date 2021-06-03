# 特征值为复数
# 确定特征值绝对值的上下界
# 确定不出来的弹出
import numpy as np
import sys


def hoskins(A, Q, delt, b=0, B=0, m=10):
    """
    该函数通过Hoskins迭代法求解Lyapunov方程：A'X+XA+Q=0;A的特征值均为
    大于零的实数，Q为对称实矩阵，也因此该方程有唯一实对称解。

    Parameters
    ----------
    A:其特征值均为大于零的实数；
    delt:指定||A-I||_1(列和范数)的范围；
    b:特征值的绝对值的正上界；
    B:特征值的绝对值的上界；
    m:尝试估计矩阵A特征值上下界的次数

    Returns
    -------
    output : 方程的近似解；
    """

    # 计算特征值的上下界
    if b == 0:
        # 通过 Gerschgorin 圆盘定理尝试确定特征值绝对值的上下界
        AA = A.copy()
        d = np.diagonal(AA)
        if np.all(d) < 0:
            AA = -AA
            A = -A
            Q = -Q
            d = -d
        elif np.any(d) <= 0:
            print('矩阵A的主对角元非全正或全负，无 \
                    法估计特征值绝对值的上下界')
            sys.exit() # 终止程序
        B = []
        k = -1
        kk = 0
        while b <= 0:
            if kk >= m:
                print('估计特征值的绝对值的上下界失败，可尝试调高尝试次数m')
                sys.exit()
            if k != -1:  # 尝试通过对角矩阵对A相似变换缩小圆盘以确定上下界
                kk += 1
                c = 0.9*d[k]/x[k]
                AA[k, :] = AA[k, :]*c
                AA[:, k] = AA[:, k]/c
            x = np.abs(AA).sum(1)-np.abs(d)
            B.append(np.max(d+x))
            b = np.min(d-x)
            k = np.argmin(d-x)
        B = np.min(B)

    I = np.eye(A.shape[0])
    while np.linalg.norm(A-I, 1) > delt:
        alpha = 2*b/(b+(b*B)**(1/2))**2
        beta = b*B*alpha
        A_ = np.linalg.inv(A)
        A = alpha*A+beta*A_
        # Construct Q = alpha*Q + beta*A_'QA_
        Q = alpha*Q+beta*(A_.T).dot(Q.dot(A_))
        epsilon = ((b-(B*b)**(1/2))/(b+(B*b)**(1/2)))**2
        b = 1-epsilon
        B = 1+epsilon   
    return -Q/2
