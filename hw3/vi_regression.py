import csv
import numpy as np
import math
from scipy.special import digamma
from scipy.stats import norm
import time
import datetime
import pickle
import matplotlib.pyplot as plt



def update_q_lambda(x_input,  y_input, e0, f0, N, mu_prime, sigma_prime):
    e_prime = e0 + (N / 2)
    f_prime = 0
    for i in range(N):
        x_i = np.matrix(x_input[i])
        f_prime += (y_input - x_i.T.dot(mu_prime))**2 + x_i.T.dot(sigma_prime).dot(x_i)
    f_prime = f_prime/2
    f_prime += f0
    return e_prime, f_prime


def update_q_alpha(D, a0, b0, mu_prime, sigma_prime, k):
    a_prime = np.zeros(shape=(D,1))
    b_prime = np.zeros(shape=(D,1))
    for k in range(D):
        a_prime[k] = a0 + .5
        b_prime[k] = b0 + (.5 * ((mu_prime[k])**2 + sigma_prime[k][k]))
    return a_prime, b_prime


def create_diag(N, a_prime, b_prime):
    diag = np.zeros(shape=(N,N))
    for i in range(N):
        diag[i][i] = a_prime[i]/b_prime[i]
    return diag


def update_q_w(x_input, y_input, a_prime, b_prime, N, e_prime, f_prime):
    sigma_prime = 0
    mu_prime = 0
    for i in range(N):
        x_i = np.matrix(x_input[i])
        sigma_prime = sigma_prime + x_i.dot(x_i.T)
        mu_prime = mu_prime + (y_input[i] * x_i)
    sigma_prime = (e_prime/f_prime)*sigma_prime
    mu_prime = (e_prime/f_prime)*mu_prime
    sigma_prime = (create_diag(N, a_prime, b_prime) + sigma_prime)
    sigma_prime = np.linalg.inv(sigma_prime)
    mu_prime = sigma_prime.dot(mu_prime)
    return sigma_prime, mu_prime


def calc_L_term_1(N, D, a_prime, b_prime, mu_prime):
    term_1 = -1 * D * math.log(2 * math.pi) / 2
    term_2 = 0
    for k in range(D):
        term_2 += digamma(a_prime[k]) - math.log(b_prime[k])
    term_2 = term_2/2
    term_3 = -0.5 * np.trace(mu_prime.dot(mu_prime.T).dot(create_diag(N, a_prime, b_prime)))
    return term_1 + term_2 + term_3





def vi_regression(data):
    a0 = b0 = 10**-16
    e0 = f0 = 1

    # a_prime = b_prime =
    # e_prime = f_prime = 1
    mu_prime = sigma_prime = 0
    N = D = 0
    q_lambda = q_alpha = q_w = 0

    # read in data
    reader = csv.reader(open("data_csv/X_set" + data + ".csv", "r"), delimiter=",")
    x = list(reader)
    x_input = np.array(x).astype("float")

    reader = csv.reader(open("data_csv/y_set" + data + ".csv", "r"), delimiter=",")
    y = list(reader)
    y_input = np.array(y).astype("float")

    reader = csv.reader(open("data_csv/z_set" + data + ".csv", "r"), delimiter=",")
    z = list(reader)
    z_input = np.array(z).astype("float")

    N = x_input.shape[0]
    D = x_input.shape[1]
    print(N)
    print(create_diag(N))


vi_regression("1")




