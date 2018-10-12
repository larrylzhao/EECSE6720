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
input_dict = {}
for line in input:
    input_dict[(line[0],line[1])] = line[2]


# initialize U and V from Normal(0, 0.1I)
U = {}
V = {}
U_old = {}
V_old = {}

for i in range(1, N+1):
    U[i] = np.random.normal(0, 0.1, d)
for j in range(1, M+1):
    V[i] = np.random.normal(0, 0.1, d)
print(U[1][0], U[1][1], U[1][2], U[1][3], U[1][4])


def EPhi(i, j):
    r_ij = input_dict(i, j)
    dot_product = np.dot(np.transpose(U_old[i]), V_old[j])
    pdf = norm.pdf(-1 * dot_product)
    cdf = norm.cdf(-1 * dot_product)
    if r_ij == 1:
        return pdf/(1-cdf) + dot_product
    else:
        return pdf/cdf + dot_product
