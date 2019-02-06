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
Xt = X.T
R = np.dot(Xt, X)

# sort eigenvalues in increasing order
evals, evecs = np.linalg.eigh(R)
idx = np.argsort(evals)[:: -1]
evecs = evecs[:, idx]

# extract the 2 dominant eigenvectors
r = 2
V_r = evecs [: ,:r]

D = np.dot(X, V_r)

# save output in comma separated filename.txt. filename depends on the script
out_file = f"{splitext(basename(sys.argv[0]))[0]}.txt"
np.savetxt (out_file, D, delimiter =',')
print(f'{out_file} is created')
