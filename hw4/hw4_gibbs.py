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
        clusters_cnt = [len(n[i]) for i in n]
        clusters_cnt.sort(reverse=True)
        if len(n.keys()) < 6:
            largest_clusters.append(clusters_cnt)
        else:
            largest_clusters.append(clusters_cnt[:6])
        clusters.append(len(n.keys()))

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

            # if c[i] in n:
            #     n[c[i]].append(i)
            # else:
            #     n[c[i]] = [i]

            try:
                n[c[i]].append(i)
            except KeyError:
                n[c[i]] = [i]

            # generate theta j
            j = len(phi) - 1
            if c[i] == j:
                theta[j] = np.random.beta(a_0 + x[i], b_0 + 20 - x[i])

            n = { k : v for k,v in n.items() if len(v) > 0}
            exist_c = list(n.keys())
            theta_n = {}
            for k in range(len(exist_c)):
                theta_n[k] = theta[exist_c[k]]
            theta = theta_n
            # Reindex clusters
            c_n = []
            n = {}
            for k in range(N):
                for j in range(len(exist_c)):
                    if c[k] == exist_c[j]:
                        c_n.append(j)
                        try:
                            n[j].append(k)
                        except KeyError:
                            n[j] = [k]
            c = c_n

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

with open('output/clusters.pkl', 'wb') as f:
    pickle.dump(clusters, f, pickle.HIGHEST_PROTOCOL)
with open('output/largest.pkl', 'wb') as f:
    pickle.dump(largest, f, pickle.HIGHEST_PROTOCOL)

with open('output/clusters.pkl', 'rb') as f:
    clusters = pickle.load(f)
with open('output/largest.pkl', 'rb') as f:
    largest = pickle.load(f)

largest_six_split = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
for i in largest:
    for j in range(len(i)):
        largest_six_split[j].append(i[j])
    if len(i) < 6:
        for k in range(len(i), 6):
            largest_six_split[k].append(0)




plt.figure(figsize=(10,10))
for i in largest_six_split:
    plt.plot(range(iterations), largest_six_split[i], label=str(i+1)+'th Largest')
plt.legend()
plt.xlabel('t')
plt.ylabel('# points in cluster')
plt.show()

plt.plot(range(iterations), clusters)
plt.xlabel('t')
plt.ylabel('number of clusters')
plt.show()

























