# Input: five input arguments:
# 1)reduced_file    2)Vector_file   3)labels_file   4)queried_point_file    5)label
# The output of the program is the row of the nearest neighbor in the reduced.txt file.
# reduced_file is a matrix of size nx2. Vector_file is a matrix of 2xm.
# labels_file is an array of size n.The labels are 1,2,3
# queried_point is an 1xm. label is an integer.
# If lable is -1, it means finding the result for reducedim1.py
# Otherwise, it means finding the result for reducedim2.py

# output file is index.txt

'''
execute
py nearest.py reducedim1_iris_reduced.txt reducedim1_iris_v.txt iris.labels query_point.txt -1
py nearest.py reducedim2_iris_reduced.txt reducedim2_iris_v.txt iris.labels query_point.txt 1
'''
import sys
import numpy as np


if len(sys.argv) != 6:
    print('usage : ', sys.argv[0], '1)reduced_file    2)Vector_file   3)labels_file   4)queried_point_file    5)label')
    sys.exit()


def findNearestPoint(test_point, point_set):
    idx = -1
    min_norm = sys.maxsize
    for i in range(len(point_set)):
        norm = np.linalg.norm(point_set[i] - test_point)
        if min_norm > norm:
            min_norm = norm
            idx = i
    return idx


def testPCA(reduced_x, bases, test_point):
    # return index of nearest neighbor

    v1 = np.reshape(bases, (2, bases.shape[1]))
    v2 = np.reshape(test_point, (test_point.shape[0], 1))
    reduced_test_point = np.reshape(np.matmul(v1, v2), (1, 2))

    return findNearestPoint(reduced_test_point, reduced_x)


def testScatter(reduced_x, bases, labels, test_point, label):
    # return index of nearest neighbor

    v1 = np.reshape(bases, (2, bases.shape[1]))
    v2 = np.reshape(test_point, (test_point.shape[0], 1))
    reduced_test_point = np.reshape(np.matmul(v1, v2), (1, 2))

    list1 = []
    list2 = []
    for i in range(len(labels)):
        if labels[i] == label:
            list1.append(reduced_x[i])
            list2.append(i)

    idx = findNearestPoint(reduced_test_point, np.matrix(list1))
    return list2[idx]

reduced_file = sys.argv[1]
vector_file = sys.argv[2]
labels_file = sys.argv[3]
queried_point_file = sys.argv[4]
label = int(sys.argv[5])



# read the files for the matrices reduced_Xt, vector, labels, queried_point and label..
reduced_x = np.genfromtxt(reduced_file, delimiter=',', autostrip=True)
bases = np.genfromtxt(vector_file, delimiter=',', autostrip=True)
labels = np.genfromtxt(labels_file, delimiter=',', autostrip=True)
test_point = np.genfromtxt(queried_point_file, delimiter=',', autostrip=True)

nn_idx = 0
if label == -1:
    nn_idx = testPCA(reduced_x, bases, test_point)
else:
    nn_idx = testScatter(reduced_x, bases, labels, test_point, label)

# save output in comma separated filename.txt.
np.savetxt("index.txt", [nn_idx], fmt='%d', delimiter=',')