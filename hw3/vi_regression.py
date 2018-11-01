import csv
import numpy as np
import math
from scipy.special import digamma, gamma
from scipy.stats import norm
import time
import datetime
import pickle
import matplotlib.pyplot as plt



def update_q_lambda(x_input,  y_input, e0, f0, N, mu_prime, sigma_prime):
    e_prime = e0 + (N / 2)
    f_prime = 0
    for i in range(N):
        x_i = np.matrix(x_input[i]).T
        f_prime += (y_input[i][0] - x_i.T.dot(mu_prime)[0,0])**2 + x_i.T.dot(sigma_prime).dot(x_i)[0,0]
    f_prime = f_prime/2
    f_prime += f0
    return e_prime, f_prime


def update_q_alpha(D, a0, b0, mu_prime, sigma_prime):
    a_prime = np.zeros(shape=(D,1))
    b_prime = np.zeros(shape=(D,1))
    for k in range(D):
        a_prime[k] = a0 + .5
        b_prime[k] = b0 + (.5 * ((mu_prime[k])**2 + sigma_prime[k,k]))
    return a_prime, b_prime


def create_diag(D, a_prime, b_prime):
    diag = np.zeros(shape=(D,D))
    for i in range(D):
        diag[i][i] = a_prime[i][0]/b_prime[i][0]
    return diag


def update_q_w(D, x_input, y_input, a_prime, b_prime, N, e_prime, f_prime):
    mu_prime = np.zeros(shape=(D,1), dtype='float64')
    sigma_prime = np.diag(np.ones(D, dtype='float64'))
    for i in range(N):
        x_i = np.matrix(x_input[i]).T
        sigma_prime = sigma_prime + x_i.dot(x_i.T)
        mu_prime = mu_prime + (y_input[i][0] * x_i)
    sigma_prime = (e_prime/f_prime)*sigma_prime
    mu_prime = (e_prime/f_prime)*mu_prime
    sigma_prime = (create_diag(D, a_prime, b_prime) + sigma_prime)
    sigma_prime = np.linalg.inv(sigma_prime)
    mu_prime = sigma_prime.dot(mu_prime)
    return mu_prime, sigma_prime


def calc_L_term_1(N, D, a_prime, b_prime, mu_prime):
    term_1 = -1 * D * math.log(2 * math.pi) / 2
    term_2 = 0
    for k in range(D):
        term_2 += digamma(a_prime[k][0]) - math.log(b_prime[k][0])
    term_2 = term_2/2
    term_3 = -0.5 * np.trace(mu_prime.dot(mu_prime.T).dot(create_diag(D, a_prime, b_prime)))
    return term_1 + term_2 + term_3


def calc_L_term_2(e0, f0, e_prime, f_prime):
    term_1 = e0 * math.log(f0)
    term_2 = -1 * math.log(gamma(e0))
    term_3 = (e0 - 1) * (digamma(e_prime) - math.log(f_prime))
    term_4 = f0 * e_prime / f_prime
    return term_1 + term_2 + term_3 + term_4


def calc_L_term_3(D, a0, b0, a_prime, b_prime):
    L_term_3 = 0
    term_1 = a0 * math.log(b0)
    term_2 = -1 * math.log(gamma(a0))
    for k in range(D):
        term_3 = (a0 - 1) * (digamma(a_prime[k][0]) - math.log(b_prime[k][0]))
        term_4 = b0 * a_prime[k][0] / b_prime[k][0]
        L_term_3 = L_term_3 + term_1 + term_2 + term_3 + term_4
    return L_term_3


def calc_L_term_4(N, e_prime, f_prime, y_input, x_input, mu_prime, sigma_prime):
    term_1 = -1 * N * math.log(2 * math.pi) / 2
    term_2 = N / 2 * (digamma(e_prime) - math.log(f_prime))
    term_3 = 0
    for i in range(N):
        x_i = np.matrix(x_input[i]).T
        term_3 = term_3 + (y_input[i][0] - x_i.T.dot(mu_prime))**2 + x_i.T.dot(sigma_prime).dot(x_i)
    term_3 = term_3 * -0.5 * e_prime / f_prime
    return term_1 + term_2 + term_3[0,0]


def calc_L_term_5(N, sigma_prime):
    term_1 = N * math.log(2 * math.pi * math.e)
    sign, term_2 = np.linalg.slogdet(sigma_prime)
    return -0.5 * (term_1 + sign * term_2)


def calc_L_term_6(e_prime, f_prime):
    term_1 = math.log(f_prime)
    term_2 = -1 * math.log(gamma(e_prime))
    term_3 = (e_prime - 1) * (digamma(e_prime) - e_prime)
    return -1 * (term_1 + term_2 + term_3)


def calc_L_term_7(D, a_prime, b_prime):
    L_term_7 = 0
    for k in range(D):
        term_1 = math.log(b_prime[k][0]) - math.log(gamma(a_prime[k][0]))
        term_2 = (a_prime[k][0] - 1) * digamma(a_prime[k][0])
        term_3 = -1 * a_prime[k][0]
        L_term_7 = L_term_7 + term_1 + term_2 + term_3
    return -1 * L_term_7


def vi_regression(data):

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

    a0 = b0 = 10**-16
    e0 = f0 = 1

    N = x_input.shape[0]
    D = x_input.shape[1]

    a_prime = b_prime = np.array([a0] * D, dtype='float64')
    e_prime = f_prime = e0
    mu_prime = np.zeros(shape=(D,1), dtype='float64')
    sigma_prime = np.diag(np.ones(D, dtype='float64'))

    L_list = []

    for i in range(50):
        e_prime, f_prime = update_q_lambda(x_input, y_input, e0, f0, N, mu_prime, sigma_prime)
        a_prime, b_prime = update_q_alpha(D, a0, b0, mu_prime, sigma_prime)
        mu_prime, sigma_prime = update_q_w(D, x_input, y_input, a_prime, b_prime, N, e_prime, f_prime)
        L_term_1 = calc_L_term_1(N, D, a_prime, b_prime, mu_prime)
        L_term_2 = calc_L_term_2(e0, f0, e_prime, f_prime)
        L_term_3 = calc_L_term_3(D, a0, b0, a_prime, b_prime)
        L_term_4 = calc_L_term_4(N, e_prime, f_prime, y_input, x_input, mu_prime, sigma_prime)
        L_term_5 = calc_L_term_5(N, sigma_prime)
        L_term_6 = calc_L_term_6(e_prime, f_prime)
        L_term_7 = calc_L_term_7(D, a_prime, b_prime)
        L = (L_term_1 + L_term_2 + L_term_3 + L_term_4 + L_term_5 + L_term_6 + L_term_7)
        print(i, L)
        L_list.append(L)

    plt.plot(range(50), L_list)
    plt.xlabel("iteration")
    plt.ylabel("variational objective function")
    plt.title("Set " + data)
    plt.show()

    plt.plot(range(D), 1/(a_prime/b_prime))
    plt.xlabel("K")
    plt.ylabel(r'$1/E_{q}[\alpha_k]$')
    plt.title("Set " + data)
    plt.show()

    print("Eq[lambda]", f_prime/e_prime)

    y_hat = x_input.dot(mu_prime)
    plt.plot(z_input, y_hat, 'r', label=r'$\hat{y}$')
    plt.scatter(z_input, y_input, c='y', label="y")
    plt.plot(z_input, 10 * np.sinc(z_input), label="ground truth")
    plt.legend()
    plt.xlabel("z")
    plt.ylabel("y")
    plt.title("Set" + data)
    plt.show()


vi_regression("1")


