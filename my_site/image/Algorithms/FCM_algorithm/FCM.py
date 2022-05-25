import numpy as np
import datetime
from dateutil.relativedelta import relativedelta


class FCM:
    def __init__(self, image, image_bit, n_clusters, m, epsilon, max_iter):
        """Modified Fuzzy C-means clustering
        <image>: 2D array, grey scale image.
        <n_clusters>: int, number of clusters/segments to create.
        <m>: float > 1, fuzziness parameter. A large <m> results in smaller
             membership values and fuzzier clusters. Commonly set to 2.
        <max_iter>: int, max number of iterations.
        """

        # -------------------Check inputs-------------------

        if np.ndim(image) != 2:
            raise Exception("<image> needs to be 2D (gray scale image).")
        if n_clusters <= 0 or n_clusters != int(n_clusters):
            raise Exception("<n_clusters> needs to be a positive integer.")
        if m < 1:
            raise Exception("<m> needs to be > 1.")
        if epsilon <= 0:
            raise Exception("<epsilon> needs to be > 0")

        self.result_def = None
        self.c = None
        self.u = None
        self.image = image
        self.image_bit = image_bit
        self.n_clusters = n_clusters  # number of clusters/segments to create
        self.m = m  # fuzziness parameter
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.shape = image.shape  # image shape
        self.X = image.flatten().astype('float')  # shape: (number of pixels)
        self.numPixels = image.size

    # ---------------------------------------------
    def initial_u(self):
        # each jth cluster (column) contains the cluster membership of the ith data point (ith row)
        # the sum of the memberships for each data point is equal to one.

        u = np.zeros((self.numPixels, self.n_clusters))
        idx = np.arange(self.numPixels)
        for ii in range(self.n_clusters):
            idxii = idx % self.n_clusters == ii
            u[idxii, ii] = 1
        return u

    def update_u(self):
        """Compute weights (cluster memberships)"""
        c_mesh, idx_mesh = np.meshgrid(self.c, self.X)  # self.c centroids of the clusters
        power = 2. / (self.m - 1)  # self.c defined in form_clusters()
        a = abs(idx_mesh - c_mesh) ** power
        b = np.sum((1. / abs(idx_mesh - c_mesh)) ** power, axis=1)

        return 1. / (a * b[:, None])

    def update_c(self):
        """Compute centroid of clusters"""  # vectorization in python to speed up the computing time
        numerator = np.dot(self.X, self.u ** self.m)
        denominator = np.sum(self.u ** self.m, axis=0)
        return numerator / denominator  # returns a matrix of shape (1,num_centroids)

    def form_clusters(self):
        """Iterative training"""
        d = 100
        self.u = self.initial_u()  # initializing the weights
        if self.max_iter != -1:
            i = 0
            self.iterate(i, True)
        else:
            i = 0
            self.iterate(i, d > self.epsilon)
        self.segmentimage()

    def iterate(self, i, condition):
        cnt = 0
        start_dt1 = datetime.datetime.now()
        while condition:
            start_dt = datetime.datetime.now()
            cnt += 1
            self.c = self.update_c()  # compute the centroids of the clusters
            old_u = np.copy(self.u)
            self.u = self.update_u()
            d = np.sum(abs(self.u - old_u))
            end_dt = datetime.datetime.now()
            diff = relativedelta(end_dt, start_dt)
            print("iter: %s, time interval: %s:%s:%s:%s" % (
                cnt, diff.hours, diff.minutes, diff.seconds, diff.microseconds))
            print("Iteration %d : cost = %f" % (i, d))

            if d < self.epsilon or i > self.max_iter:
                print('Converge at iteration {}'.format(cnt))
                end_dt1 = datetime.datetime.now()
                diff = relativedelta(end_dt1, start_dt1)
                print("duration time: %s:%s:%s:%s" % (diff.hours, diff.minutes, diff.seconds, diff.microseconds))
                break
            i += 1

    def defuzzify(self):
        return np.argmax(self.u, axis=1)  # Returns the indices of the maximum values along an axis.
        # returns the max membership value of each data point

    def segmentimage(self):
        """Segment image based on max weights"""

        result_def = self.defuzzify()
        self.result_def = (result_def.reshape(self.shape).astype('int'))*255

        return self.result_def
