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


#deprecated
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


def update_phi_alt(n, K, a, b, alpha, x):
    phi = np.empty((n, K))
    t1 = []
    t2 = []
    t3 = []
    for k in range(K):
        t1.append(phi_t1(a[k], b[k]))
        t2.append(phi_t2(a[k], b[k]))
        t3.append(phi_t3(k, alpha))
    for i in range(n):
        denom = 0
        for k in range(K):
            denom += np.exp(x[i]*t1[k] + (20-x[i])*t2[k] + t3[k])
        for j in range(K):
            num = np.exp(x[i]*t1[j] + (20-x[i])*t2[j] + t3[j])
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
        t3 += gammaln(a_0+b_0) - gammaln(a_0) - gammaln(b_0) + (a_0-1)*phi_t1(a[j], b[j]) + (b_0-1)*phi_t2(a[j], b[j])
    return t3


def calc_L_t_4(alpha_0, K, alpha):
    temp1 = gammaln(sum(alpha_0))
    temp2 = K * gammaln(alpha_0[0])
    temp3 = digamma(sum(alpha))
    t4 = temp1 - temp2
    for i in range(K):
        t4 += alpha_0[i] * (digamma(alpha[i]) - digamma(temp3))
    return t4


def calc_L_t_5(x, n, K, phi):
    t2 = 0
    for i in range(n):
        for j in range(K):
            t2 += phi[i,j] * math.log(phi[i,j])
    return t2


def calc_L_t_6(K, alpha):
    temp1 = gammaln(sum(alpha))
    temp2 = 0
    for i in range(K):
        temp2 += gammaln(alpha[i])
    t4 = temp1 - temp2
    for i in range(K):
        t4 += (alpha[i]-1) * digamma(alpha[i])
    t4 -= (sum(alpha) - K) * digamma(sum(alpha))
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
    L = {}
    for i in range(7):
        L[i] = []

    for t in range(iterations):
        print(t)
        phi = update_phi_alt(n, K, a, b, alpha, x)
        # print(phi)
        # phi_alt = update_phi_alt(n, K, a, b, alpha, x)
        # if np.array_equal(phi, phi_alt):
        #     print("not equal!!")
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
        L[0].append(t1)
        L[1].append(t2)
        L[2].append(t3)
        L[3].append(t4)
        L[4].append(t5)
        L[5].append(t6)
        L[6].append(t7)
    return L, phi

x = fetch_data()
n = x.shape[0]
iterations = 1000
K = 3
L, phi3 = vi(K, iterations)
# plot_scatter(x, n, K, phi3)
with open('output/L3.pkl', 'wb') as f:
    pickle.dump(L, f, pickle.HIGHEST_PROTOCOL)
with open('output/L3.pkl', 'rb') as f:
    L3 = pickle.load(f)
L3_total = []
for i in range(len(L3[4])):
    L3_total.append(L3[0][i]+L3[1][i]+L3[2][i]+L3[3][i]-L3[4][i]-L3[5][i]-L3[6][i])

K = 15
L, phi15 = vi(K, iterations)
# plot_scatter(x, n, K, phi3)
with open('output/L15.pkl', 'wb') as f:
    pickle.dump(L, f, pickle.HIGHEST_PROTOCOL)
with open('output/L15.pkl', 'rb') as f:
    L15 = pickle.load(f)
L15_total = []
for i in range(len(L15[4])):
    L15_total.append(L15[0][i]+L15[1][i]+L15[2][i]+L15[3][i]-L15[4][i]-L15[5][i]-L15[6][i])

K = 50
L, phi50 = vi(K, iterations)
# plot_scatter(x, n, K, phi3)
with open('output/L50.pkl', 'wb') as f:
    pickle.dump(L, f, pickle.HIGHEST_PROTOCOL)
with open('output/L50.pkl', 'rb') as f:
    L50 = pickle.load(f)
L50_total = []
for i in range(len(L50[4])):
    L50_total.append(L50[0][i]+L50[1][i]+L50[2][i]+L50[3][i]-L50[4][i]-L50[5][i]-L50[6][i])

start = 1
plt.plot(range(start,iterations), L3_total[start:iterations], label="K=3")
plt.plot(range(start,iterations), L15_total[start:iterations], label="K=15")
plt.plot(range(start,iterations), L50_total[start:iterations], label="K=50")
plt.xlabel('t')
plt.ylabel('Log Likelihood')
plt.title('Objective Function')
plt.legend()
plt.show()
