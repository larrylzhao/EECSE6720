import csv
import numpy as np
import math
from scipy.stats import binom
from scipy.special import digamma, gammaln
import pickle
import matplotlib.pyplot as plt


# read in data
def fetch_data():
    reader = csv.reader(open("x.csv", "r"), delimiter=",")
    x = list(reader)
    x = [i[0] for i in x]
    x_input = np.array(x).astype("int")
    return x_input


def q_c_t1(a, b):
    return digamma(a) - digamma(a+b)


def q_c_t2(a, b):
    return digamma(b) - digamma(a+b)


def q_c_t3(j, alpha):
    alpha_sum = 0
    for k in range(alpha.shape[0]):
        alpha_sum += alpha[k]
    return digamma(alpha[j]) - digamma(alpha_sum)


def update_q_c(n, K, a, b, alpha, x):
    q_c = np.empty((n, K))
    for i in range(n):
        for j in range(K):
            t1 = q_c_t1(a[j], b[j])
            t2 = q_c_t2(a[j], b[j])
            t3 = q_c_t3(j, alpha)
            num = np.exp(x[i]*t1 + (20-x[i])*t2 + t3)
            denom = 0
            for k in range(K):
                tk3 = q_c_t3(k, alpha)
                denom += np.exp(x[i]*t1 + (20-x[i])*t2 + tk3)
            q_c[i,j] = num / denom
    return q_c


def set_n_j(q_c, n, K):
    n_j = np.zeros(K)
    for j in range(K):
        for i in range(n):
            n_j[j] += q_c[i][j]
    return n_j


def update_q_pi(alpha_0, n_j):
    return alpha_0 + n_j


def update_q_theta(a_0, q_c, x, b_0, n, K):
    a = [a_0]*K
    b = [b_0]*K
    for j in range(K):
        for i in range(n):
            a[j] += q_c[i,j] * x[i]
            b[j] += q_c[i,j] * (20-x[i])
    return a, b


def calc_L_t_1(n, K, q_c, x, a, b):
    t1 = 0
    for i in range(n):
        for j in range(K):
            t1 += q_c[i,j] * (x[i] * q_c_t1(a[j], b[j]) + (20-x[i]) * q_c_t2(a[j], b[j]))
    return t1


def calc_L_t_2():
    temp1 = 0
    temp2 = 0
    temp3 = 0
    temp4 = 0
    t2 = 0
    for i in range(n):
        for j in range(K):



def vi(K, iterations):
    x = fetch_data()
    n = x.shape[0]
    alpha_0 = np.ones(K)
    alpha_0 = alpha_0 / 10
    alpha = alpha_0
    a_0 = 0.5
    a = [a_0]*K
    b_0 = 0.5
    b = [b_0]*K
    theta = np.random.rand(K)
    L = []

    for t in range(iterations):
        print(t)
        q_c = update_q_c(n, K, a, b, alpha, x)
        n_j = set_n_j(q_c, n, K)
        alpha = update_q_pi(alpha_0, n_j)
        a, b = update_q_theta(a_0, q_c, x, b_0, n, K)


vi(3, 100)
