import csv
import numpy as np
from scipy.stats import norm
import time
import datetime
import pickle

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


U = {}
V = {}
U_old = {}
V_old = {}
EPhi = {}

# initialize U and V from Normal(0, 0.1I)
for i in range(1, N+1):
    U[i] = np.matrix(np.random.normal(0, 0.1, d)).T
for j in range(1, M+1):
    V[j] = np.matrix(np.random.normal(0, 0.1, d)).T

identity = np.identity(d, dtype=float)
def update_U():
    for i in range(1, N+1):
        term_1 = identity
        term_2 = np.zeros((d, 1), dtype=float)
        for j in input_dict_u[i].keys():
            term_1 = term_1 + V[j].dot(V[j].T)
            term_2 = term_2 + V[j]*EPhi[(i, j)]
        U[i] = np.linalg.inv(term_1).dot(term_2)

def update_V():
    for j in range(1, M+1):
        if j in input_dict_v.keys():
            term_1 = identity
            term_2 = np.zeros((d, 1), dtype=float)
            for i in input_dict_v[j].keys():
                term_1 = term_1 + U[i].dot(U[i].T)
                term_2 = term_2 + U[i]*EPhi[(i, j)]
            V[j] = np.linalg.inv(term_1).dot(term_2)


def update_EPhi():
    for i in range(1, N+1):
        for j in input_dict_u[i].keys():
            r_ij = input_dict_u[i][j]
            # dot_product = np.dot(np.transpose(U_old[i]), V_old[j])
            dot_product = U_old[i].T.dot(V_old[j])[0, 0]
            pdf = norm.pdf(-1 * dot_product)
            cdf = norm.cdf(-1 * dot_product)
            if r_ij == 1:
                EPhi[(i, j)] = (pdf/(1-cdf) + dot_product)
            else:
                EPhi[(i, j)] = (-1 * pdf/cdf + dot_product)


dln = d / 2 * np.log(1 / 2 / np.pi / c)
LNP = 0.0
def update_LNP():
    term_1 = 0.0
    term_2 = 0.0
    term_3 = 0.0
    for i in range(1, N+1):
        term_1 += dln - ( (U[i].T.dot(U[i]))[0, 0] / 2 )
    for j in range(1, M+1):
        term_2 += dln - ( (V[j].T.dot(V[j]))[0, 0] / 2 )
    for i in range(1, N+1):
        for j in input_dict_u[i].keys():
            r_ij = input_dict_u[i][j]
            if r_ij == 1:
                term_3 += np.log(norm.cdf((U[i].T.dot(V[j]))[0, 0]))
            else:
                term_3 += np.log(1 - norm.cdf((U[i].T.dot(V[j]))[0, 0]))
    # print(term_1, term_2, term_3)
    return term_1 + term_2 + term_3

def getTime():
    ts = time.time()
    return "[" + datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') + "] "


f_lnp = open("output/lnnp_1.txt", "w")

print(dln)
print("***************************************")

for t in range(1, 101):
    U_old = dict(U)
    V_old = dict(V)
    print(getTime() + "iteration " + str(t))

    print(getTime() + "updating EPhi ")
    update_EPhi()
    print(getTime() + "updating U ")
    update_U()
    print(getTime() + "updating V ")
    update_V()
    print(getTime() + "updating LNP ")
    LNP = update_LNP()
    print(getTime() + "LNP: " + str(LNP))
    f_lnp.write(str(LNP) + "\n")
    print("***************************************")


# save U, V, and EPhi
with open('output/U.pkl', 'wb') as f:
    pickle.dump(U, f, pickle.HIGHEST_PROTOCOL)
with open('output/V.pkl', 'wb') as f:
    pickle.dump(U, f, pickle.HIGHEST_PROTOCOL)
with open('output/EPhi.pkl', 'wb') as f:
    pickle.dump(U, f, pickle.HIGHEST_PROTOCOL)

U_tmp = {}
with open('output/U.pkl', 'rb') as f:
    U_tmp = pickle.load(f)


# evaluate test data
true_pos = true_neg = false_pos = false_neg = 0

# open files into numpy arrays
reader = csv.reader(open("ratings_test.csv", "r"), delimiter=",")
x = list(reader)
test_input = np.array(x).astype("int")

for entry in test_input:
    i = entry[0]
    j = entry[1]
    r = entry[2]

    pr1 = np.log(norm.cdf((U[i].T.dot(V[j]))[0, 0]))
    pr0 = np.log(1 - norm.cdf((U[i].T.dot(V[j]))[0, 0]))

    if pr1 > pr0:
        if r == 1:
            true_pos += 1
        else:
            false_pos += 1
    else:
        if r == 1:
            false_neg += 1
        else:
            true_neg += 1

print(true_pos, true_neg, false_pos, false_neg)



