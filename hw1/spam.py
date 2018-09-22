import csv
import numpy as np
import math
# import matpyplot


def calc_prior(e, f, cnt, N):
    return (e + cnt) / (N + e + f)


# this one has overflow issues
def calc_pred_distr(N, a, b, X, x_new):
    part1 = ((N + b) / (N + b + 1))
    part1 = part1 ** (X + a)
    part2 = (1 / (N + 1 + b)) ** x_new
    constant = math.factorial(X + x_new + a - 1) / (math.factorial(x_new) * math.factorial(X + a - 1))
    print(part1, part2, constant)
    return part1 * part2 * constant


def calc_pred_distr2(N, a, b, X, x_new):
    part1 = (X + a) * math.log10((N + b) / (N + b + 1))
    part2 = x_new * math.log10((1 / (N + 1 + b)))
    constant = math.log10(math.factorial(X + x_new + a - 1)) - math.log10(math.factorial(x_new)) - math.log10(math.factorial(X + a - 1))
    # print(part1, part2, constant)
    return 10 ** (part1 + part2 + constant)


# open files into numpy arrays
reader = csv.reader(open("X_train.csv", "r"), delimiter=",")
x = list(reader)
X_train = np.array(x).astype("int")

reader = csv.reader(open("label_train.csv", "r"), delimiter=",")
x = list(reader)
label_train = np.array(x).astype("int")

reader = csv.reader(open("X_test.csv", "r"), delimiter=",")
x = list(reader)
X_test = np.array(x).astype("int")

reader = csv.reader(open("label_test.csv", "r"), delimiter=",")
x = list(reader)
label_test = np.array(x).astype("int")


N = X_train.shape[0]
feature_cnt = X_train.shape[1]
a = 1
b = 1
e = 1
f = 1

spam_cnt_train = np.count_nonzero(label_train)
nonspam_cnt_train = N - spam_cnt_train
spam_cnt_test = np.count_nonzero(label_test)
nonspam_cnt_test = label_test.shape[0] - spam_cnt_test

spam_train = X_train[0:spam_cnt_train]
nonspam_train = X_train[spam_cnt_train:N]



# calculate the priors
prior_spam = calc_prior(e, f, spam_cnt_train, N)
prior_nonspam = calc_prior(e, f, nonspam_cnt_train, N)

print(prior_spam)
print(prior_nonspam)
# calculate the predictive distributions

spam_X_sum = spam_train.sum(axis=0)
nonspam_X_sum = nonspam_train.sum(axis=0)
print(spam_X_sum)


true_pos = true_neg = false_pos = false_neg = 0

for i in range(label_test.shape[0]):
    p_spam = prior_spam
    p_nonspam = prior_nonspam
    for feature in range(feature_cnt):
        p_spam *= calc_pred_distr2(spam_cnt_train, a, b, spam_X_sum[feature], X_test[i][feature])
        p_nonspam *= calc_pred_distr2(nonspam_cnt_train, a, b, nonspam_X_sum[feature], X_test[i][feature])

    p_actual_spam = 0
    if (p_spam == 0) and (p_nonspam == 0):
        p_actual_spam = .5
    else:
        p_actual_spam = p_spam / (p_spam + p_nonspam)
    print(i+1, "\t", p_spam, "\t", p_nonspam, "\t", p_actual_spam)

    result = "nonspam"
    correct = False
    if p_actual_spam > .5:
        result = "spam"
    actual = label_test[i]

    if result == "spam":
        if actual == 1:
            true_pos += 1
        else:
            false_pos += 1
    else:
        if actual == 1:
            false_neg += 1
            print("***********************************")
        else:
            true_neg += 1

print(true_pos, true_neg, false_pos, false_neg)
