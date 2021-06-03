import numpy as np


def tridiagonalization(AA, M=1e-3):
    """
    通过Geist在二十世纪九十年代初提出的三对角化方法将矩阵分解：A=R^{-1}SR,其中R是
    三对角矩阵；该方法具有较好的数值稳定性的优点。

    Parameters
    ----------
    AA : n阶实方阵
        需要三对角化的矩阵
    M : float,optioal
        一个乘子的最大允许上界

    Returns
    -------
    S : 三对角矩阵
    R : 可逆矩阵
        满足A=R^{-1}SR
    """

    A = AA.copy()
    n = A.shape[0]
    R = np.eye(n)
    while True:
        tol = M  # 乘子的最大允许上界
        x = False  # 表示是否进行了重新开始的修复
        c = 0  # 因乘子大于 tol 而连续进行 LR 迭代修复(简称：迭代修复）的次数
        d = -1  # 上次修复时的 k
        m = 0  # 进行 LR 迭代修复时的起始行索引
        k = 0
        while k < n-2:
            if tol > 10*M:  # 扩大十倍后连续修复3次依然不行
                A, R = renovate(AA, n)  # 进行重新开始的修复
                x = True  # 重新开始
                break
            # 开始对A进行约化
            v = A[k+1:, k]
            w = A[k, k+1:]
            delt = v.dot(w)
            if delt == 0:
                if np.all(w == 0):
                    m = k+1
                    if np.any(v != 0):
                        p = np.argmax(np.abs(v))
                        interchange(A, R, k, p)
                        lstrans(A, R, k)
                else:
                    if np.all(v == 0):
                        q = np.argmax(np.abs(w))
                        interchange(A, R, k, q)
                        ustrans(A, R, k)
                    else:  # w,v均不为零向量，进行修复
                        A, R = fixup(A, R, m, k)
                        continue  # 重新执行该次循环
            else:
                p, omega = pivot(v, w, delt)
                if omega > tol:  # 乘子的最大模大于最大允许上界，进行修复
                    if k == d:  # 连续修复
                        c += 1
                        if c > 3:  # 已连续修复3次
                            tol = 10*tol  # 将上界扩大 10 倍
                            c = 0  # 再给三次机会
                            continue
                    else:
                        c = 1
                        d = k  # 更新修复时的行索引
                    A, R = fixup(A, R, m, k)
                    continue  # 重新执行该次循环
                else:
                    interchange(A, R, k, p)
                    lstrans(A, R, k)
                    ustrans(A, R, k)
            k += 1
        if not(x):  # 若未进行重新开始的修复，计算完毕
            return A, R  # A即为最终的三对角矩阵S


def renovate(A, n):
    """
    对n阶矩阵 A进行重新开始的修复，R^{-1}AR，这里 R 选择 Householder 矩阵

    Returns
    -------
    A : 修复后的 A
    R : R=I-2uu'(u为n维单位向量) 
    """
    A = A.copy()
    u = np.random.rand(n)*2-1  # -1~1之间均匀随机分布的数组
    u = u/np.linalg.norm(u)  # 单位化
    # Construct I-2uu'( u 视为列向量)
    R = np.eye(n)-2*np.outer(u, u)  # Householder 矩阵
    A = R.dot(A.dot(R))
    return A, R


def interchange(A, R, k, p):
    """确定了主元所在的位置后，对A的行列进行相应的交换，并更新R"""
    A[[k+1, k+p+1], k:] = A[[k+p+1, k+1], k:]
    A[k:, [k+1, k+p+1]] = A[k:, [k+p+1, k+1]]
    R[[k+1, k+p+1], :] = R[[k+p+1, k+1], :]


def lstrans(A, R, k):
    """确定了初等下三角矩阵，并对A进行相似变换，然后更新R"""
    l = A[k+2:, k]/A[k+1, k]
    A[k+2:, k] = 0
    A[k+2:, k+1:] = A[k+2:, k+1:]-np.outer(l, A[k+1, k+1:])
    A[k:, k+1] = A[k:, k+1]+A[k:, k+2:].dot(l)
    R[k+2:, :] = R[k+2:, :]-np.outer(l, R[k+1, :])


def ustrans(A, R, k):
    """确定了初等上三角矩阵，并对A进行相似变换，然后更新R"""
    u = A[k, k+2:]/A[k, k+1]
    A[k, k+2:] = 0
    A[k+1:, k+2:] = A[k+1:, k+2:]-np.outer(A[k+1:, k+1], u)
    # 此时A[k+2:,k]已全为零
    A[k+1, k+1:] = A[k+1, k+1:]+u.dot(A[k+2:, k+1:])
    R[k+1, :] = R[k+1, :]+u.dot(R[k+2:, :])


def fixup(AA, RR, m, k):
    """当消去过程中断时，对A进行单位移的隐式LR迭代修复，并且更新R"""
    while True:  # 除数为零时重新开始（可考虑指定除数的范围）
        A = AA.copy()
        R = RR.copy()
        s = np.random.uniform(0.1, 1, 1)
        if k == m:
            A[m:, m+1] = A[m:, m+1]-s*A[m:, m]
            A[m, m:] = A[m, m:]+s*A[m+1, m:]
            R[m, :] = R[m, :]+s*R[m+1, :]
            return A, R
        A[m, m] = A[m, m]+s*A[m+1, m]
        A[m, m+1] = A[m, m+1]+s*(A[m+1, m+1]-A[m, m])
        A[m, m+2] = s*A[m+1, m+2]
        A[m+1, m+1] = A[m+1, m+1]-s*A[m+1, m]
        R[m, :] = R[m, :]+s*R[m+1, :]
        if k > m+1:
            for i in range(m+1, k):
                if A[i-1, i] == 0:
                    print('除数为零')
                    continue
                r = A[i-1, i+1]/A[i-1, i]
                A[i-1, i+1] = 0
                A[i, i] = A[i, i]+r*A[i+1, i]
                A[i, i+1] = A[i, i+1]+r*(A[i+1, i+1]-A[i, i])
                A[i, i+2] = r*A[i+1, i+2]
                A[i+1, i+1] = A[i+1, i+1]-r*A[i+1, i]
                R[i, :] = R[i, :]+r*R[i+1, :]
        if k == m+1:
            r = s
        A[k-1, k+2:] = r*A[k, k+2]
        if A[k-1, k] == 0:
            print('除数为零')
            continue
        u = A[k-1, k+1:]/A[k-1, k]  # 除数有可能为零
        A[k-1, k+1:] = 0
        A[k:, k+1:] = A[k:, k+1:]-np.outer(A[k:, k], u)
        A[k, k:] = A[k, k:]+u.dot(A[k+1:, k:])
        R[k, :] = R[k, :]+u.dot(R[k+1:, :])
        return A, R


def pivot(v, w, delt):
    """确定列主元的在v上的索引，以及相应的乘子的最大模"""
    v1 = np.max(np.abs(v))
    r = np.argmax(np.abs(v))
    w1 = np.max(np.abs(w))
    q = np.argmax(np.abs(w))
    # 确定v,w的模第二大的分量
    vv = v.copy()
    vv[r] = 0
    v2 = np.max(np.abs(vv))
    ww = w.copy()
    ww[q] = 0
    w2 = np.max(np.abs(ww))
    omega = np.inf
    for i in range(v.size):
        v0 = np.abs(v[i])
        if v[i] == 0:
            continue
        if i == r:
            nu = v2/v0
        else:
            nu = v1/v0
        if i == q:
            mu = w2*v0/np.abs(delt)
        else:
            mu = w1*v0/np.abs(delt)
        gamma = np.abs(w[i]*v[i]/delt)
        t = max(nu, mu, gamma)
        if t < omega:
            omega = t
            p = i
    return p, omega
