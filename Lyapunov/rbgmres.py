import numpy as np


def rbgmres(A, Q, epsilon, m, X=None):
    """
    重新开始的GMRES算法求解方程 A'X+XA+Q=0

    Parameters
    ----------
    A,Q : 均为n阶方阵
    epsilon : float
        精度要求，||A'X+XA+Q||_F<=epsilon
    m : int
        希望使用的Krylov子空间的维数
    X : n阶方阵,optional
        方程解的迭代初值
        Defaults : 零矩阵

    Returns
    -------
    X : 满足精度要求的近似解
    """
    
    if X is None:
        X = np.zeros(A.shape)
    R = A.T.dot(X)+X.dot(A)+Q
    beta = np.linalg.norm(R, 'fro')
    k=0
    while beta > epsilon and k<2:
        V1 = R/beta
        V, H = arnoldi(A, V1, m)
        y, beta = least_squares_method(H, beta)
        for i in range(y.size):
            X += y[i]*V[i]
        R = A.T.dot(X)+X.dot(A)+Q
        k+=1
    print(k)
    return(X)


def arnoldi(A, V1, m):
    """
    至多计算m个矩阵V2,V3,...,Vm+1和m(m+1)/2+m个数h_ij s.t. tr(Vi^H,Vj)=delt_ij,
    并且A^T Vj+VjA=sum_k=1^j+1(h_kj*Vk)

    Parameters
    ----------
    A : n*n矩阵
    V1 : n*n矩阵，并且||V1||_F=1
    m : 期望计算的矩阵的个数

    Returns
    -------
    V : list
        V[i]=Vi+1,i=0,1,...,m-1
    H : 矩阵，H[i,j]=h_ij(i>=j+1);H[i,j]=0(其它)
    """
    
    V = []
    V.append(V1)
    H = np.zeros((m + 1, m))
    for j in range(m):
        W = A.T.dot(V[j]) + V[j].dot(A)
        for i in range(j+1):
            H[i, j] = np.trace(W.T.dot(V[i]))
            W = W - H[i, j] * V[i]
        H[j + 1, j] = np.linalg.norm(W, 'fro')
        if H[j + 1, j] < 0.01:  # Krylov序列在第j+1项终止
            H = H[:j, :j - 1]  # j+1*j阶
            break
        if j != m-1:
            V.append(W / H[j + 1, j])
    return (V, H)


def least_squares_method(A, beta):
    """
    通过Givens变换求A的QR分解，然后求解 ||Ax+beta*e_1||_2 的最小二乘问题

    Parameters
    ----------
    A : (n+1)*n阶Hessenberg矩阵
    beta : float

    Returns
    -------
    x : n维数组
        最小二乘问题的解
    beta : float
        最小二乘问题的最小值
    """

    # 通过Givens变换对A进行QR分解
    n = A.shape[1]
    I = np.eye(n+1)
    Q = I.copy()
    for i in range(n):
        # 构造初等旋转矩阵
        a = np.linalg.norm(A[i:i+2, i])
        c = A[i, i]/a
        s = A[i+1, i]/a
        T = I.copy()
        T[np.ix_([i, i+1], [i, i+1])] = np.array([[c, s], [-s, c]])
        # 更新A,Q
        A = T.dot(A)
        Q = Q.dot(T.T)
    x = -beta*np.linalg.inv(A[:n, :]).dot(Q[0, :n])
    beta = beta*np.abs(Q[0, n])
    return x, beta
