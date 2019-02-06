import sys
import numpy as np
from os.path import basename, splitext

# py reducedim1.py iris.data iris.labels

if len(sys.argv) != 3:
    print('usage : ', sys.argv[0], 'data_file labels_file ')
    sys.exit()

data_argv = sys.argv[1]

# read in data
x = np.genfromtxt(data_argv, delimiter=',', autostrip=True)
x_trans = x
x = np.transpose(x)

# compute x * x_trans
R = np.matmul(x, x_trans)

# conmpute eigenvalues, eigenvectors
evals, evecs = np.linalg.eigh(R)
idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:, idx]

# compute the 2 bases, the result after pca
bases = evecs[:, :2]
bases_trans = np.transpose(bases)
pca_result = np.matmul(bases_trans, x)

# save output
script_name = splitext(basename(sys.argv[0]))[0]
data_name = splitext(basename(data_argv))[0]

output_vec = f'{script_name}_{data_name}_v.txt'
output_data = f'{script_name}_{data_name}_reduced.txt'
np.savetxt(output_vec, bases_trans, delimiter=',')
np.savetxt(output_data, np.transpose(pca_result),  delimiter=',')

