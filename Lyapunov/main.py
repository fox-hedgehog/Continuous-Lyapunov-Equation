import numpy as np
import scipy as sp

from vec import vec
from bs_lyap import bs_lyap
from hammarling import hammarling
from hoskins import hoskins
from ADI import ADI
from conjugate_gradient import conjugate_gradient as c_g
from rbgmres import rbgmres


# A = np.array([[4, 3, 2], [3, 5, 4], [2, 4, 6]])  # 正定矩阵
Q = np.array([[2, 3, 1], [3, 2, 4], [1, 4, 3.0]])  # 对称矩阵
A = np.array([[8, 2, 3],
               [3, 9, 4],
               [2,4,7.0]])  # 主对角占优矩阵
X_m = np.array([[0.00155039, -0.51162791, 0.26434109],
                [-0.51162791, 0.5503876, -0.55426357],
                [0.26434109, -0.55426357, 0.03139535]])  # A^TX+XA+Q=0的解
X_0 = np.array([[0.00155039, -0.51162791, 0.26434109],
                [-0.51162791, 1, -0.55426357],
                [1, -0.55426357, 0.03139535]])   # 近似解
eps = 1e-5
km = 5
m = 4

# X=vec(A.T,A,-Q)
# X=bs_lyap(A,Q)
# X=hoskins(A,Q,eps)
# X = ADI(-A, -Q, km)
# X=c_g(A,Q,eps,km)
# X=rbgmres(A,Q,eps,m)

# print(X)
# print(A.T.dot(X)+X.dot(A)+Q)
# print(np.linalg.norm(A.T.dot(X)+X.dot(A)+Q,'fro'))
# print(A1.T.dot(X)+X.dot(A1)+Q)

# 验证 Hammarling 程序
A2=-A # 稳定矩阵
C=np.array([[1,2,3],[3,4,5]]) # 满足(C,A)可观测
# M=np.vstack([np.vstack([C,C.dot(A2)]),C.dot(A2.dot(A2))])
# print(np.linalg.matrix_rank(M))  %矩阵的阶数

G=hammarling(A2,C)
# print(G)
# print(G.T.dot(G))
# X=vec(A2,A2.T,-C.T.dot(C))
# print(sp.linalg.cholesky(X))
# print(np.linalg.eig(X))
print(np.linalg.norm(A2.dot(G.T.dot(G))+G.T.dot(G.dot(A2.T))+C.T.dot(C),'fro'))
