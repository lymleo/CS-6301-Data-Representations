import sys
import numpy as np
from os.path import basename, splitext


if len(sys.argv) != 3:
    print('usage : ', sys.argv[0], 'data_file labels_file ')
    sys.exit()

data_file, label_file = sys.argv[1], sys.argv[2]

X = np.genfromtxt(data_file, delimiter=',', autostrip=True)
Y = np.genfromtxt(label_file, autostrip=True)
n, m = X.shape

W1 = np.zeros((m,m))
W2 = np.zeros((m,m))
W3 = np.zeros((m,m))
V= np.zeros((m,m))

s1 = np.zeros((1,m))
s2 = np.zeros((1,m))
s3 = np.zeros((1,m))
s = np.zeros((1,m))

m1 = 0
m2 = 0
m3 = 0
m = 0


for i, x in enumerate (X):
    if Y[i] == 1:
        s1 += x
        m1 += 1
    if Y[i] == 2:
        s2 += x
        m2 += 1
    if Y[i] == 3:
        s3 += x
        m3 += 1

mu1 = s1/m1
mu2 = s2/m2
mu3 = s3/m3

for i, x in enumerate(X):
    s += x
    m += 1
mu = s/m

B = np.dot((mu - mu1).T,mu - mu1) * m1
B += np.dot((mu - mu2).T, mu - mu2) * m2
B += np.dot((mu - mu3).T, mu - mu3) * m3

for i, x in enumerate(X):
    if Y[i] == 1:
        W1 += np.dot((x-mu1).T, x-mu1)
    if Y[i] == 2:
        W1 += np.dot((x-mu2).T, x-mu2)
    if Y[i] == 3:
        W1 += np.dot((x-mu3).T, x-mu3)

W = W1 + W2 + W3
W_inv = np.linalg.inv(W)

C = np.dot(W_inv, B)

w, e = np.linalg.eigh(C)
idx = np.argsort(w)[:: -1]
e = e[:,idx]
w = w[idx]
r = 2
V_r = e[:,:r]

D = np.dot(X, V_r)




# save output in comma separated filename.txt. filename depends on the script
out_file = f"{splitext(basename(sys.argv[0]))[0]}.txt"
np.savetxt (out_file, D, delimiter =',')
print(f'{out_file} is created')
