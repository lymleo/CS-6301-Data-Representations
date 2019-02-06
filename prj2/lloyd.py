# Input: k (#clusters), r (#iterations), datafile
# k is an integer. r is an integer. Each row in datafile is a point. 
# Output: cluster number of each data points

import sys
import numpy as np
from numpy import newaxis
from numpy.random import randint

np.seterr(divide='ignore', invalid='ignore')

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

# initial selection
def initialize_centroids(points, k):
    """returns k centroids from the initial points"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]

centroids = initialize_centroids(points, k)


def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)


def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

for i in range(r):
    closest = closest_centroid(points, centroids)
    centroids = move_centroids(points, closest, centroids)


def compute_quantization_error(points, centroids, closest):
    m = 0
    for i in range(len(points)):
        m += ((points[i] - centroids[closest[i]])**2).sum()
    print(f"Quantitative Error is: {m}")
    return m
compute_quantization_error(points, centroids, closest)


np.savetxt(output, closest, delimiter=',', fmt = '%i')
