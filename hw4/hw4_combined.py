import csv
import numpy as np
import math
from scipy.stats import binom
from scipy.misc import comb
from scipy.special import digamma, gammaln, beta as B
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

##########################################################################
# 4.1
def em(K, iterations):
    x = fetch_data()
    n = x.shape[0]
    pi = np.ones(K)
    theta = np.random.rand(K)
    f = []

    for t in range(iterations):
        print (t)
        if t == 14:
            print("hit 14")
        phi = np.zeros((n, K))
        for i in range(n):
            for j in range(K):
                denom = 0
                for k in range(K):
                    pmf = binom.pmf(x[i], 20, theta[k])
                    if math.isnan(pmf):
                        pmf = 0
                    denom += pi[k] * pmf
                pmf = binom.pmf(x[i], 20, theta[j])
                if math.isnan(pmf):
                    pmf = 0
                num = pi[j] * pmf
                phi[i, j] = num / denom
        # print(phi)
        for j in range(K):
            n_j = 0
            theta[j] = 0
            for i in range(n):
                n_j += phi[i,j]
                theta[j] += phi[i,j] * x[i] / 20
            theta[j] = theta[j] / n_j
            pi[j] = n_j / n

        f_t = 0
        for i in range(n):
            for j in range(K):
                f_t += phi[i,j] * np.log(binom.pmf(x[i], 20, theta[j]) * pi[j] / phi[i,j])
        f.append(f_t)

    return f, phi


def plot_scatter(x, n, K, phi):
    best = [0]*n
    for i in range(n):
        max = 0
        for j in range(K):
            if phi[i][j] > max:
                max = phi[i][j]
                best[i] = j
    plt.scatter(x, best)
    plt.xlabel('integer')
    plt.ylabel('cluster')
    plt.title('Most Probable Cluster')
    plt.show()


iterations = 50
K = 3
x = fetch_data()
n = x.shape[0]
f3, phi3 = em(K, iterations)
plot_scatter(x, n, K, phi3)
K = 9
f9, phi9 = em(K, iterations)
plot_scatter(x, n, K, phi9)
K = 15
f15, phi15 = em(K, iterations)
plot_scatter(x, n, K, phi15)

# with open('output/f.pkl', 'wb') as f:
#     pickle.dump(f, f, pickle.HIGHEST_PROTOCOL)
# with open('output/phi3.pkl', 'wb') as f:
#     pickle.dump(phi3, f, pickle.HIGHEST_PROTOCOL)
#
# with open('output/f.pkl', 'rb') as f:
#     f2 = pickle.load(f)
# with open('output/phi3.pkl', 'rb') as f:
#     phi3 = pickle.load(f)
#
# print(f)

plt.plot(range(1,iterations), f3[1:], label="K=3")
plt.plot(range(1,iterations), f9[1:], label="K=9")
plt.plot(range(1,iterations), f15[1:], label="K=15")
plt.xlabel('t')
plt.ylabel('Log Likelihood')
plt.title('Objective Function')
plt.legend()
plt.show()

##########################################################################

##########################################################################
# 4.2

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

##########################################################################

##########################################################################
# 4.3


def gibbs(iterations):
    x = fetch_data()
    N = x.shape[0]

    a_0 = b_0 = 0.5
    alpha_0 = 0.75
    c = [0] * N

    # initialize 30 clusters to start
    n = {}
    theta = {}
    for i in range(30):
        n[i] = []
        theta[i] = np.random.beta(a_0, b_0)
    # place all points in one cluster to begin
    n[0] = list(range(N))

    # precalculate the phis
    p_c = []
    Bab = B(a_0,b_0)
    for i in range(N):
        p_c.append(alpha_0 / (alpha_0 + N - 1) * comb(20, x[i]) * B(x[i] + a_0, 20 - x[i] + b_0) / Bab)

    clusters = []
    largest_clusters = []

    for t in range(iterations):
        print(t)
        clusters.append(len(n.keys()))

        clusters_cnt = []
        for i in n:
            clusters_cnt.append(len(n[i]))
        clusters_cnt.sort(reverse=True)
        length = 6
        if len(n.keys()) < 6:
            length = len(n.keys())
        largest_clusters.append(clusters_cnt[:length])

        for i in range(N):
            phi = []
            for clust in n:
                if c[i] == clust:
                    n[clust].remove(i)
                phi.append(binom.pmf(x[i], 20, theta[clust]) * len(n[clust]) / (alpha_0 + N - 1))
            phi.append(p_c[i])

            # normalize phi
            phi_sum = sum(phi)
            for k in range(len(phi)):
                phi[k] = phi[k]/phi_sum

            # sample the index c[i] from a discrete distribution with this parameter
            c[i] = int(np.random.choice(len(phi), 1, p=phi))

            if c[i] in n:
                n[c[i]].append(i)
            else:
                n[c[i]] = [i]

            # generate theta j
            j = len(phi) - 1
            if c[i] == j:
                theta[j] = np.random.beta(a_0 + x[i], b_0 + 20 - x[i])

            c_positive = []
            for clust, points in n.items():
                if len(points) > 0:
                    c_positive.append(clust)
            n = {}
            theta_temp = {}
            for k in range(len(c_positive)):
                theta_temp[k] = theta[c_positive[k]]
            theta = theta_temp
            c_temp = []
            for k in range(N):
                for j in range(len(c_positive)):
                    if c[k] == c_positive[j]:
                        c_temp.append(j)
                        if j in n:
                            n[j].append(k)
                        else:
                            n[j] = [k]
            c = c_temp

        print(str(t) + " 2")
        for j in n:
            a = a_0
            b = b_0
            for i in range(N):
                if c[i] == j:
                    a += x[i]
                    b += 20 - x[i]
            theta[j] = np.random.beta(a, b)


    return clusters, largest_clusters


iterations = 1000
clusters, largest = gibbs(iterations)

with open('output/clusters2.pkl', 'wb') as f:
    pickle.dump(clusters, f, pickle.HIGHEST_PROTOCOL)
with open('output/largest2.pkl', 'wb') as f:
    pickle.dump(largest, f, pickle.HIGHEST_PROTOCOL)

with open('output/clusters2.pkl', 'rb') as f:
    clusters = pickle.load(f)
with open('output/largest2.pkl', 'rb') as f:
    largest = pickle.load(f)

largest_six = [[], [], [], [], [], []]
for i in largest:
    for j in range(len(i)):
        largest_six[j].append(i[j])
    if len(i) < 6:
        for k in range(len(i), 6):
            largest_six[k].append(0)


for i in range(len(largest_six)):
    plt.plot(range(iterations), largest_six[i])
plt.xlabel('t')
plt.ylabel('# points in cluster')
plt.show()

plt.plot(range(iterations), clusters)
plt.xlabel('t')
plt.ylabel('number of clusters')
plt.show()

##########################################################################























