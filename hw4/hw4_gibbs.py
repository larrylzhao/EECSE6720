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
    n = {0: list(range(N))}
    a_0 = b_0 = 0.5
    alpha_0 = 0.75
    c = [1] * N
    theta = [np.random.beta(a_0, b_0)]

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
            for iter in range(len(phi)):
                phi[iter] = phi[iter]/phi_sum

            # sample the index c[i] from a discrete distribution with this parameter
            c[i] = int(np.random.choice(len(phi), 1, p=phi))

            if c[i] in n:
                n[c[i]].append(i)
            else:
                n[c[i]] = [i]

gibbs(10)
