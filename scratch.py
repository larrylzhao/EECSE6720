import matplotlib.pyplot as plt
import numpy as np

mu_prime = np.ones(shape=(3,1), dtype='float64')
temp = 5 * mu_prime
print(temp)

# with open("hw2/output/lnnp_1.txt") as f:
#     lines1 = f.read().splitlines()
#     lines1 = [float(i) for i in lines1[:99]]
#
# with open("hw2/output/lnnp_2.txt") as f:
#     lines2 = f.read().splitlines()
#     lines2 = [float(i) for i in lines2[:99]]
#
# with open("hw2/output/lnnp_3.txt") as f:
#     lines3 = f.read().splitlines()
#     lines3 = [float(i) for i in lines3[:99]]
#
# with open("hw2/output/lnnp_4.txt") as f:
#     lines4 = f.read().splitlines()
#     lines4 = [float(i) for i in lines4[:99]]
#
# with open("hw2/output/lnnp_5.txt") as f:
#     lines5 = f.read().splitlines()
#     lines5 = [float(i) for i in lines5[:99]]
#
#
# x = np.array(list(range(20, 100)))
# plt.plot(x, lines1[18:98], label="run 1")
# plt.plot(x, lines2[18:98], label="run 2")
# plt.plot(x, lines3[18:98], label="run 3")
# plt.plot(x, lines4[18:98], label="run 4")
# plt.plot(x, lines5[18:98], label="run 5")
# plt.xlabel("iteration")
# plt.ylabel("lnp(R, U, V)")
# plt.legend()
# plt.show()