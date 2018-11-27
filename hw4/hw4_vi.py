import csv
import numpy as np
import math
from scipy.stats import binom
from scipy.misc import comb
from scipy.special import digamma, gammaln
import pickle
import matplotlib.pyplot as plt
np.random.seed(0)


# read in data
def fetch_data():
    reader = csv.reader(open("x.csv", "r"), delimiter=",")
    x = list(reader)
    x = [i[0] for i in x]
    x_input = np.array(x).astype("int")
    return x_input


def phi_t1(a, b):
    return digamma(a) - digamma(a+b)


def phi_t2(a, b):
    return digamma(b) - digamma(a+b)


def phi_t3(j, alpha):
    alpha_sum = 0
    for k in range(alpha.shape[0]):
        alpha_sum += alpha[k]
    return digamma(alpha[j]) - digamma(alpha_sum)


def update_phi(n, K, a, b, alpha, x):
    phi = np.empty((n, K))
    for i in range(n):
        for j in range(K):
            t1 = phi_t1(a[j], b[j])
            t2 = phi_t2(a[j], b[j])
            t3 = phi_t3(j, alpha)
            num = np.exp(x[i]*t1 + (20-x[i])*t2 + t3)
            denom = 0
            for k in range(K):
                tk1 = phi_t1(a[k], b[k])
                tk2 = phi_t2(a[k], b[k])
                tk3 = phi_t3(k, alpha)
                denom += np.exp(x[i]*tk1 + (20-x[i])*tk2 + tk3)
            phi[i,j] = num / denom
    return phi


def set_n_j(phi, n, K):
    n_j = np.zeros(K)
    for j in range(K):
        for i in range(n):
            n_j[j] += phi[i,j]
    return n_j


def update_q_pi(alpha_0, n_j):
    return alpha_0 + n_j


def update_q_theta(a_0, phi, x, b_0, n, K):
    a = [a_0]*K
    b = [b_0]*K
    for j in range(K):
        for i in range(n):
            a[j] += phi[i,j] * x[i]
            b[j] += phi[i,j] * (20-x[i])
    return a, b


def calc_L_t_1(n, K, phi, x, a, b):
    t1 = 0
    for i in range(n):
        for j in range(K):
            t1 += phi[i,j] * (x[i] * phi_t1(a[j], b[j]) + (20-x[i]) * phi_t2(a[j], b[j]))
    return t1


def calc_L_t_2(n, K, phi, alpha):
    t2 = 0
    for i in range(n):
        for j in range(K):
            t2 += phi[i,j] * (digamma(alpha[j]) - digamma(sum(alpha)))
    return t2


def calc_L_t_3(K, a_0, b_0, a, b):
    t3 = 0
    for j in range(K):
        t3 += gammaln(a_0+b_0) - gammaln(a_0) - gammaln(b_0) + (a_0-1) * phi_t1(a[j], b[j]) + (b_0-1) * phi_t2(a[j], b[j])
    return t3


def calc_L_t_4(alpha_0, K, alpha):
    temp1 = gammaln(alpha_0[0] * K)
    temp2 = K * gammaln(alpha_0[0])
    temp3 = digamma(sum(alpha))
    t4 = temp1 - temp2
    for i in range(K):
        t4 += alpha_0[i] * (digamma(alpha[i]) - digamma(temp3))
    return t4


def calc_L_t_5(x, n, K, phi):
    t2 = math.log(math.factorial(n))
    for i in range(n):
        for j in range(K):
            t2 += n * phi[i,j] * math.log(phi[i,j])
            t2 -= comb(K, x[i]) * (phi[i,j] ** x[i]) * (1 - phi[i,j] ** (K-x[i]) * math.log(math.factorial(x[i])))
    return t2


def calc_L_t_6(K, alpha):
    temp1 = gammaln(sum(alpha))
    temp2 = 0
    for i in range(K):
        temp2 += gammaln(alpha[i])
    temp3 = digamma(sum(alpha))
    t4 = temp1 - temp2
    for i in range(K):
        t4 += alpha[i] * (digamma(alpha[i]) - digamma(temp3))
    return t4


def calc_L_t_7(K, a, b):
    t3 = 0
    for j in range(K):
        t3 += gammaln(a[j]+b[j]) - gammaln(a[j]) - gammaln(b[j]) + (a[j]-1) * phi_t1(a[j], b[j]) + (b[j]-1) * phi_t2(a[j], b[j])
    return t3


def plot_scatter(x, n, K, phi):
    best = [0]*n
    for i in range(n):
        max = 0
        for j in range(K):
            if phi[i,j] > max:
                max = phi[i,j]
                best[i] = j
    plt.scatter(x, best)
    plt.xlabel('integer')
    plt.ylabel('cluster')
    plt.title('Most Probable Cluster')
    plt.show()


def vi(K, iterations):
    x = fetch_data()
    n = x.shape[0]
    alpha_0 = np.ones(K)
    alpha_0 = alpha_0 / 10
    alpha = np.random.rand(K)
    a_0 = 0.5
    a = [a_0]*K
    # a = np.random.rand(K)
    b_0 = 0.5
    b = [b_0]*K
    # b = np.random.rand(K)
    L = []

    for t in range(iterations):
        print(t)
        phi = update_phi(n, K, a, b, alpha, x)
        print(phi)
        n_j = set_n_j(phi, n, K)
        alpha = update_q_pi(alpha_0, n_j)
        a, b = update_q_theta(a_0, phi, x, b_0, n, K)

        t1 = calc_L_t_1(n, K, phi, x, a, b)
        t2 = calc_L_t_2(n, K, phi, alpha)
        t3 = calc_L_t_3(K, a_0, b_0, a, b)
        t4 = calc_L_t_4(alpha_0, K, alpha)
        t5 = calc_L_t_5(x, n, K, phi)
        t6 = calc_L_t_6(K, alpha)
        t7 = calc_L_t_7(K, a, b)
        L.append(t1 + t2 + t3 + t4 - t5 - t6 - t7)
    return L, phi

x = fetch_data()
n = x.shape[0]
iterations = 1000
K = 15
L3, phi3 = vi(K, iterations)
plot_scatter(x, n, K, phi3)

plt.plot(range(1,iterations), L3[1:], label="K=3")
plt.xlabel('t')
plt.ylabel('Log Likelihood')
plt.title('Objective Function')
plt.legend()
plt.show()
