import numpy as np
import os
import random
import sys


# python3 lloyd.py iris.data 3 20 irisL.txt
if len(sys.argv) != 5:
    print('usage: ', sys.argv[0], 'data_file k r output')
    # print('usage: ', sys.argv[0], 'k r data_file')
    sys.exit()

#Read inputs.
data_file = sys.argv[1]
k = int(sys.argv[2])
r = int(sys.argv[3])
# customize output file
output = sys.argv[4]
with open(data_file,"r") as filestream:
	points = np.loadtxt(filestream, delimiter=",")


def compute_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

def compute_squared_distance(point, centroid):
    return np.sum((point - centroid)**2)

def compute_new_cij(i, j, points, centroids):
    k = len(centroids)
    sum = 0
    for t in range(0, k):
        sum += compute_squared_distance(points[i], centroids[j]) / compute_squared_distance(points[i], centroids[t])
    return 1 / sum


def iterate_c_means(points, centroids, r):
    for iteration in range(0, r):
        total_points = len(points)
        k = len(centroids)
        n, m = points.shape
        c0 = np.zeros((total_points, k))
        c = np.zeros((total_points, k))
        for index_point in range(0, total_points):
            distance_sum_in_cluster = 0.0
            for index_centroid in range(0, k):
                if compute_euclidean_distance(points[index_point], centroids[index_centroid]) == 0:
                    c0[index_point][index_centroid] = 0
                else:
                    c0[index_point][index_centroid] = 1 / compute_euclidean_distance(points[index_point], centroids[index_centroid])
                distance_sum_in_cluster += c0[index_point][index_centroid]
            for index_centroid in range(0, k):
                c[index_point][index_centroid] = c0[index_point][index_centroid] / distance_sum_in_cluster
        for index_centroid in range(0, k):
            sum1 = np.zeros(m)
            sum2 = 0
            for index_point in range(0, total_points):
                sum1 += (c[index_point][index_centroid] **2) * points[index_point]
                sum2 += (c[index_point][index_centroid] **2)
            centroids[index_centroid] = sum1 / sum2

        # update of cij
        for index_point in range(0, total_points):
            for index_centroid in range(0, k):
                c[index_point][index_centroid] = compute_new_cij(index_point, index_centroid, points, centroids)

    # compute quantitation error
    m = 0.0
    for index_point in range(0, total_points):
        for index_centroid in range(0, k):
            m += (c[index_point][index_centroid] **2) * compute_squared_distance(points[index_point], centroids[index_centroid])
            #m += compute_squared_distance(points[index_point], centroids[index_centroid])
            #print("m = ", m)
    print("Quantization Error: ", m, "\n")


    # convert soft clustering to hard clustering and output the label
    cluster_label = []
    for index_point in range(0, total_points):
        max_membershipvalue = -9999.0
        max_index = -1
        for index_centroid in range(0, k):
            if max_membershipvalue < c[index_point][index_centroid]:
                max_membershipvalue = c[index_point][index_centroid]
                max_index = index_centroid
        cluster_label.append(max_index)
    return [cluster_label, centroids]


def create_centroids(points):
    centroids = []

    data_row, data_column = points.shape

    arr = np.arange(data_row)
    np.random.shuffle(arr)
    for i in range(0, k):
        centroids.append(points[arr[i]])
    return np.array(centroids)


centroids = create_centroids(points)
cluster_label, centroids = iterate_c_means(points, centroids, r)
cluster_label = np.array(cluster_label)
# print(cluster_label)
np.savetxt(output, cluster_label, delimiter=',', fmt = '%i')


