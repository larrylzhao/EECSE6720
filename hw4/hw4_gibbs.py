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
        n[i] = list(range(N))
        theta[i] = np.random.beta(a_0, b_0)

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
# clusters, largest = gibbs(iterations)
#
# with open('output/clusters.pkl', 'wb') as f:
#     pickle.dump(clusters, f, pickle.HIGHEST_PROTOCOL)
# with open('output/largest.pkl', 'wb') as f:
#     pickle.dump(largest, f, pickle.HIGHEST_PROTOCOL)

with open('output/clusters.pkl', 'rb') as f:
    clusters = pickle.load(f)
with open('output/largest.pkl', 'rb') as f:
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

























