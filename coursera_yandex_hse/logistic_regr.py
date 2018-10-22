from sklearn.metrics import roc_auc_score
import numpy as np
from sys import exit
from sklearn.metrics import classification_report

print("[INFO] loading data")
with open('data-logistic.csv', 'r') as f:
    data = f.read()

data = data.split('\n')

X, X1, X2 = [], [], []
Y = []
for sample in data:
    if sample == '':
        continue
    Y.append(int(sample.split(',')[0]))
    X1.append([float(x) for x in sample.split(',')][1])
    X2.append([float(x) for x in sample.split(',')][2])

l = len(X1)
eps = 1e-5
count = 0
k = 0.1
w1 = 0
w2 = 0
# without regularization
C = 0
#w1 = w1_0 + k/l*np.sum([ (y*x1)/(1+np.exp(-y*(w1_0*x1+w2_0*x2))) for y, x1, x2 in zip(y, X1, X2)]) - k*C*w1_0
#w2 = w2_0 + k/l*np.sum([ (y*x2)/(1+np.exp(-y*(w1_0*x1+w2_0*x2))) for y, x1, x2 in zip(y, X1, X2)]) - k*C*w2_0
print("[INFO] gradient descent without regularization")
while True:
    w1_new = w1 + k*np.mean([ Y[i]*X1[i]*(1 - 1.0/(1 + np.exp(-Y[i]*(w1*X1[i]+w2*X2[i])))) for i in range(l)])
    w2_new = w2 + k*np.mean([ Y[i]*X2[i]*(1 - 1.0/(1 + np.exp(-Y[i]*(w1*X1[i]+w2*X2[i])))) for i in range(l)])
    count += 1
    if abs(w1_new - w1) < eps and abs(w2_new - w2) < eps or count > 10000:
        break
    w1, w2 = w1_new, w2_new

print(count)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


print(w1, w2)
y_ = [1 if sigmoid(w1*x1+w2*x2) >= 0.5 else -1 for x1, x2 in zip(X1, X2) ]
no_reg = round(roc_auc_score(Y, [sigmoid(w1*X1[i] + w2*X2[i]) for i in range(l)]), 3)
no_reg_ = round(roc_auc_score(Y, y_), 3)
print(classification_report(Y, y_))
print("[INFO] gradient descent with regularization")
# with regularization

w1, w2 = 0, 0
C = 10
count = 0
while True:
    w1_new = w1 + k*np.mean([Y[i]*X1[i]*(1 - 1.0/(1 + np.exp(-Y[i]*(w1*X1[i]+w2*X2[i])))) for i in range(l)]) - k*C*w1
    w2_new = w2 + k*np.mean([Y[i]*X2[i]*(1 - 1.0/(1 + np.exp(-Y[i]*(w1*X1[i]+w2*X2[i])))) for i in range(l)]) - k*C*w2
    count += 1
    if abs(w1_new - w1) < eps and abs(w2_new - w2) < eps or count > 10000:
        break
    w1, w2 = w1_new, w2_new

print(count)
print(w1, w2)
with_reg = round(roc_auc_score(Y, [sigmoid(w1*X1[i] + w2*X2[i]) for i in range(l)]), 3)
y_ = [1 if sigmoid(w1*x1 + w2*x2) >= 0.5 else -1 for x1, x2 in zip(X1, X2) ]
print(classification_report(Y, y_))
with open('file.txt', 'w') as f:
    f.write('{} {}'.format(no_reg, with_reg))

print(no_reg, with_reg)
w1_ = 0.28801877
w2_ = 0.09179177
w1_C = 0.02855938
w2_C = 0.02478083
print(round(roc_auc_score(Y, [sigmoid(w1_*X1[i] + w2_*X2[i]) for i in range(l)]), 3))
print(round(roc_auc_score(Y, [sigmoid(w1_C*X1[i] + w2_C*X2[i]) for i in range(l)]), 3))

