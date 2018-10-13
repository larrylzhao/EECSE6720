import csv
import numpy as np
from scipy.stats import norm

d=5
c=1
covar=1

# open files into numpy arrays
reader = csv.reader(open("ratings.csv", "r"), delimiter=",")
x = list(reader)
input = np.array(x).astype("int")


# get N and M
N, M, r_max = input.max(axis=0)


# convert input data to a dict
input_dict_u = {}
input_dict_v = {}
for line in input:
    if line[0] not in input_dict_u.keys():
        input_dict_u[line[0]] = {}
    input_dict_u[line[0]][line[1]] = line[2]
    if line[1] not in input_dict_v.keys():
        input_dict_v[line[1]] = {}
    input_dict_v[line[1]][line[0]] = line[2]


# initialize U and V from Normal(0, 0.1I)
U = {}
V = {}
U_old = {}
V_old = {}

for i in range(1, N+1):
    U[i] = np.matrix(np.random.normal(0, 0.1, d)).T
for j in range(1, M+1):
    V[i] = np.matrix(np.random.normal(0, 0.1, d)).T


Ephi = {}

def EPhi(i, j):
    r_ij = input_dict_u[i][j]
    dot_product = np.dot(np.transpose(U_old[i]), V_old[j])
    pdf = norm.pdf(-1 * dot_product)
    cdf = norm.cdf(-1 * dot_product)
    if r_ij == 1:
        return pdf/(1-cdf) + dot_product
    else:
        return pdf/cdf + dot_product

identity = np.identity(d, dtype=float)

def update_U():
    for i in range(1, N+1):
        term_1 = identity
        term_2 = np.zeros((d, 1), dtype=float)
        for j in input_dict_u[i].keys():
            term_1 = term_1 + V[j].dot(V[j].T)
            term_2 = term_2 + V[j]*Ephi[(i, j)]
        U[i] = np.linalg.inv(term_1).dot(term_2)

def update_V():
    for j in range(1, M+1):
        term_1 = identity
        term_2 = np.zeros((d, 1), dtype=float)
        for i in input_dict_v[j].keys():
            term_1 = term_1 + U[i].dot(U[i].T)
            term_2 = term_2 + U[i]*Ephi[(i, j)]
        V[j] = np.linalg.inv(term_1).dot(term_2)


update_U()