import numpy as np

d = 5

u = np.matrix(np.random.normal(0, 0.1, d)).T

identity = np.identity(d, dtype=float)
term_1 = np.linalg.inv(identity + u.dot(u.T))
term_2 = u*2
total = term_1.dot(term_2)

# print(term_1)
# print()
print(total)

print(np.linalg.inv(term_1))

temp = np.zeros((d, d), dtype=float)

print(temp)

print(u)

print(u.T)

print(u)