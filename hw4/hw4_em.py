import csv
import numpy as np
import math
from scipy.stats import binom
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
        print(phi)
        # n_j = np.zeros(n)
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
    plt.show()


iterations = 50
K = 9
f, phi = em(K, iterations)

# with open('output/f.pkl', 'wb') as f:
#     pickle.dump(f, f, pickle.HIGHEST_PROTOCOL)
# with open('output/phi.pkl', 'wb') as f:
#     pickle.dump(phi, f, pickle.HIGHEST_PROTOCOL)
#
# with open('output/f.pkl', 'rb') as f:
#     f2 = pickle.load(f)
# with open('output/phi.pkl', 'rb') as f:
#     phi2 = pickle.load(f)
#
# print(f)

plt.plot(range(1,iterations), f[1:])
plt.xlabel('t')
plt.ylabel('Log Likelihood')
plt.title('Objective Function')
plt.show()


print(phi)
x = fetch_data()
n = x.shape[0]
plot_scatter(x, n, K, phi)

