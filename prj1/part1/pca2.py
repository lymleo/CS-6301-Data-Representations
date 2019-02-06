import sys
import numpy as np
from os.path import basename, splitext


if len(sys.argv) != 3:
    print('usage : ', sys.argv[0], 'data_file labels_file ')
    sys.exit()

data_file, label_file = sys.argv[1], sys.argv[2]

X = np.genfromtxt(data_file, delimiter=',', autostrip=True)
Y = np.genfromtxt(label_file, autostrip=True)

# transpose X
X = X.T
mu = np.mean(X, axis = 1)
Xc = X.T - mu.T
C = np.dot(Xc.T, Xc)

# sort eigenvalues in increasing order
evals, evecs = np.linalg.eigh(C)
idx = np.argsort(evals)[:: -1]
evecs = evecs[:, idx]

# extract the 2 dominant eigenvectors
r = 2
V_r = evecs [: ,:r]

D = np.dot(Xc, V_r)

# save output in comma separated filename.txt. filename depends on the script
out_file = f"{splitext(basename(sys.argv[0]))[0]}.txt"
np.savetxt (out_file, D, delimiter =',')
print(f'{out_file} is created')
