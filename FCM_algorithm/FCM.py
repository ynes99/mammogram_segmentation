import numpy as np


class FCM:
    def __init__(self, image, image_bit, n_clusters, m, epsilon, max_iter):
        """Fuzzy C-means clustering
        <image>: 2D array, image en niveaux de gris.
        <n_clusters>: int, nombre de clusters/segments à créer.
        <m>: float > 1, parametre flou (fuzziness parameter). un grand <m> entraîne une plus petite
        valeur d'appartenance et des clusters plus flous. Généralement défini à 2.
        <max_iter>: int, nombre d'itérations maximales.
        """

        # -------------------Check inputs-------------------
        if np.ndim(image) != 2:
            raise Exception("<image> doit étre 2D (gray scale image).")
        if n_clusters <= 0 or n_clusters != int(n_clusters):
            raise Exception("<n_clusters> doit étre positive.")
        if m < 1:
            raise Exception("<m> est supérieur à 1.")
        if epsilon <= 0:
            raise Exception("<epsilon> est strictement positive")

        self.image = image
        self.image_bit = image_bit
        self.n_clusters = n_clusters  # nombre de clusters/segments à créer
        self.m = m  # fuzziness parameter
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.shape = image.shape  # image shape
        self.X = image.flatten().astype('float')  # shape: (number of pixels)
        self.numPixels = image.size

    # ---------------------------------------------
    def initial_u(self):
        """Initialisation de la matrice d'appartenance ,chaque jéme colonne (segment/cluster) contient la valeur
        d'appartenance de la iéme ligne (iéme datapoint) la somme des valeurs d'appartenance d'un segment est égale à 1
        each jth cluster (column) contains the cluster membership of the ith data point (ith row)
        the sum of the memberships for each segment (column) is equal to one."""

        u = np.zeros((self.numPixels, self.n_clusters))
        idx = np.arange(self.numPixels)
        for ii in range(self.n_clusters):
            idxii = idx % self.n_clusters == ii
            u[idxii, ii] = 1
        return u

    def update_u(self):
        """Compute weights/calculer les poids (cluster memberships/valeur d'appartenance)"""
        c_mesh, idx_mesh = np.meshgrid(self.c, self.X)  # self.c centres des segments
        power = 2. / (self.m - 1)  # self.c définie dans form_clusters()
        a = abs(idx_mesh - c_mesh) ** power
        b = np.sum((1. / abs(idx_mesh - c_mesh)) ** power, axis=1)

        return 1. / (a * b[:, None])

    def update_c(self):
        """Compute centroid of clusters/calculer les centre des clusters"""
        numerator = np.dot(self.X, self.u ** self.m)
        denominator = np.sum(self.u ** self.m, axis=0)
        return numerator / denominator

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
        while condition:
            self.c = self.update_c()  # compute the centroids of the clusters
            old_u = np.copy(self.u)
            self.u = self.update_u()
            d = np.sum(abs(self.u - old_u))
            print("Iteration %d : cost = %f" % (i, d))

            if d < self.epsilon or i > self.max_iter:
                break
            i += 1

    def defuzzify(self):
        return np.argmax(self.u, axis=1)
        # retourne la valeur d'appartenance maximale de chaque point

    def segmentimage(self):
        """Segmenter l'image en se basant sur les poids"""

        result = self.defuzzify()
        self.result = result.reshape(self.shape).astype('int')

        return self.result
