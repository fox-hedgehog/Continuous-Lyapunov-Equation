import numpy as np


def c_gammas(A, k, M=1e-3):
    """
    计算A对应的交替方向法(ADI)所需要的k个参数

    Parameters
    ----------
    A : 实n阶稳定矩阵
    k : 正整数
    M : float,optional
        删除列表中近似元素时所允许的误差范围

    Returns
    -------
    gamma : k维数组
    """

    n = A.shape[0]

    gamma = np.zeros(k, A.dtype)  # 存储生成的 k 个参数
    # 应用 Aronoldi 方法生成 A 的近似特征值数组 E0
    k_ = np.ceil(k/2)
    K0 = np.array([])
    k0 = np.random.randint(k_, k+1)
    kt = True
    while K0.size < k0:
        d = k0-K0.size
        if d >= n:
            t = n
        elif kt:
            t = d
            kt = False
        T = c_arnoldi(A, t)
        T = T[np.real(T) < 0]
        K0 = np.hstack((K0, T))
    K1 = np.array([])
    k1 = np.random.randint(k_, k)
    A_ = np.linalg.inv(A)  # A的逆
    kt = True
    while K1.size < k1:
        d = k1-K1.size
        if d > n:
            t = n
        elif kt:
            t = d
            kt = False
        T = c_arnoldi(A_, t)
        T = T[np.real(T) < 0]
        K1 = np.hstack((K1, T))
    E0 = np.hstack((K0, 1/K1))

    n0 = E0.size
    alphas = np.zeros(n0, E0.dtype)  # 极小极大问题中的临时变量
    for i in range(n0):
        alphas[i] = np.max(np.abs((E0-E0[i])/(E0+E0[i])))
    kk = np.argmin(alphas)
    alpha = E0[kk]
    E = np.delete(E0, kk)  # 删除已选出的元素
    gamma[0] = alpha
    if k==1:
        return gamma
    m = 0  # 已赋值的 gamma 索引
    if np.imag(alpha) != 0:
        gamma[1] = np.conjugate(alpha)
        m = 1
        if np.any((E-gamma[m])<M):
            E = np.delete(E, np.argmin(np.abs((E-gamma[m]))))  # 删除选中的近似元素

    T = np.zeros(n0)  # 计算极小极大问题中使用的临时变量
    while m < k-1:
        alphas = np.zeros(E.size)
        for i in range(E.size):
            for j in range(n0):
                t = np.abs((gamma[:m+1]-E0[j])/(gamma[:m+1]+E0[j])).prod()
                T[j] = np.abs((E0[j]-E[i])/(E0[j]+E[i]))*t
            alphas[i] = np.max(T)
        kk = np.argmin(alphas)
        alpha = E[kk]
        E = np.delete(E, kk)
        gamma[m+1] = alpha
        m += 1
        if m==k-1:
                break
        if np.imag(alpha) != 0:
            gamma[m+1] = np.conjugate(alpha)
            m += 1
            if np.any((E-gamma[m])<M):
                E = np.delete(E, np.argmin(np.abs(E-gamma[m])))
    return gamma


# 如下算法可参考《矩阵计算六讲》[徐树方，钱江] p170
def c_arnoldi(A, k, v=None):
    """用经典Arnoldi算法计算n阶实矩阵A的k(k<=n)个Ritz值"""

    n = A.shape[0]
    if v is None:
        v = np.random.uniform(-1, 1, n)
    q1 = v/np.linalg.norm(v, 2)  # 单位化
    while True:
        H = boarnoldi(A, q1, k)[0]  # 进行长度为 k 的 Arnoldi 分解
        d = np.diagonal(H)
        if d.size == k:
            break
        else:
            v = np.random.uniform(-1, 1, n)
            q1 = v/np.linalg.norm(v, 2)
    return d


def boarnoldi(A, q, k):
    """
    用重正交化Arnoldi算法计算一个长度为k的Arnoldi分解

    Parameters
    ----------
    A : n阶实矩阵
    q : n维单位向量
    k : 正整数，(k<=n)

    Reruens
    -------
    Q : n*(k+1)阶矩阵
    H : (k+1)*k阶矩阵
    """

    n = A.shape[0]
    if A.dtype=='complex64' or A.dtype=='complex128':
        dt=A.dtype
    else:
        dt='float64'
    
    Q = np.zeros((n, k+1), dt)
    Q[:, 0] = q
    H = np.zeros((k+1, k), dt)
    for j in range(k):
        omega = A.dot(Q[:, j])
        for i in range(j+1):
            H[i, j] = np.dot(Q[:, i], omega)
            omega -= H[i, j]*Q[:, i]
        for i in range(j+1):  # 重正交化
            s = np.dot(Q[:, i], omega)
            H[i, j] = H[i, j]+s
            omega -= s*Q[:, i]
        H[j+1, j] = np.linalg.norm(omega, 2)
        if H[j+1, j] == 0:
            H = H[:j, :j-1]
            Q = Q[:, :j]
            print("只计算了长度为 %d 的Arnoldi分解" % (j-1))
            break
        else:
            Q[:, j+1] = omega/H[j+1, j]
    return H, Q
