# maximize the ratio of between-class scatter and within-class scatter.

# py reducedim2.py iris.data iris.labels

import sys
import numpy as np
from os.path import basename, splitext
# py reducedim2.py iris.data iris.labels


if len(sys.argv) != 3:
    print('usage : ', sys.argv[0], 'data_file labels_file ')
    sys.exit()

data_argv = sys.argv[1]
label_argv = sys.argv[2]
x = np.genfromtxt(data_argv, delimiter=',', autostrip=True)
y = np.genfromtxt(label_argv, delimiter=',', autostrip=True)

i = 0
l1 = []
l2 = []
l3 = []
while i < len(y):
    if y[i] == 1:
        l1.append(x[i])
    elif y[i] == 2:
        l2.append(x[i])
    else:
        l3.append(x[i])
    i += 1

x1 = np.array(l1)
x2 = np.array(l2)
x3 = np.array(l3)

x1_trans = x1 - x1.mean(0)
x1 = np.transpose(x1_trans)
x2_trans = x2 - x2.mean(0)
x2 = np.transpose(x2_trans)
x3_trans = x3 - x3.mean(0)
x3 = np.transpose(x3_trans)

dim = x1_trans.shape[1]
# compute S1
# Sj = sum((y−μj)(y−μj)T)
S1 = np.zeros([dim, dim], dtype=float)
i = 0
while i < x1.shape[1]:
    v1 = np.reshape(x1[:, i], (dim, 1))
    v2 = np.reshape(x1_trans[i], (1, dim))
    s0 = np.matmul(v1, v2)
    S1 += s0
    i += 1

# compute S2
S2 = np.zeros([dim, dim], dtype=float)
i = 0
while i < x2.shape[1]:
    v1 = np.reshape(x2[:, i], (dim, 1))
    v2 = np.reshape(x2_trans[i], (1, dim))
    s0 = np.matmul(v1, v2)
    S2 += s0
    i += 1

# compute S3
S3 = np.zeros([dim, dim], dtype=float)
i = 0
while i < x3.shape[1]:
    v1 = np.reshape(x3[:, i], (dim, 1))
    v2 = np.reshape(x3_trans[i], (1, dim))
    s0 = np.matmul(v1, v2)
    S3 += s0
    i += 1

# W =  sum(Sj)
W = S1 + S2 + S3

# compute B
m1 = x1.shape[1]
m2 = x2.shape[1]
m3 = x3.shape[1]
mean1 = x1.mean(1)
mean2 = x2.mean(1)
mean3 = x3.mean(1)
mean = (m1 * mean1 + m2 * mean2 + m3 * mean3) / (m1 + m2 + m3)

dim = mean.shape[0]
# B =  mj (μj − μ)(μj − μ)T
vec1 = np.reshape(mean1 - mean, (dim, 1))
vec2 = np.reshape(np.transpose(mean1 - mean), (1, dim))
B1 = m1 * np.dot(vec1, vec2)

vec1 = np.reshape(mean2 - mean, (dim, 1))
vec2 = np.reshape(np.transpose(mean2 - mean), (1, dim))
B2 = m2 * np.dot(vec1, vec2)

vec1 = np.reshape(mean3 - mean, (dim, 1))
vec2 = np.reshape(np.transpose(mean3 - mean), (1, dim))
B3 = m3 * np.dot(vec1, vec2)

B = B1 + B2 + B3


#C=W_inv B
C = np.dot(np.linalg.inv(W), B)

# compute the evd of B
# conmpute eigenvalues, eigenvectors
evals, evecs = np.linalg.eig(B)
idx = np.argsort(evals)[::-1]
evals = np.real(evals)[idx]
evecs = np.real(evecs)[:, idx]

# compute the 1 bases, the result after pca
bases = evecs[:, :2]
bases_trans = np.transpose(bases)
pca_result = np.matmul(bases_trans, np.transpose(x))

# save output
script_name = splitext(basename(sys.argv[0]))[0]
data_name = splitext(basename(data_argv))[0]
output_vec = f'{script_name}_{data_name}_v.txt'
output_data = f'{script_name}_{data_name}_reduced.txt'
np.savetxt(output_vec, bases_trans, delimiter=',')
np.savetxt(output_data, np.transpose(pca_result), delimiter=',')
