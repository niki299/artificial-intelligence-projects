from __future__ import print_function
import numpy, random, copy


class Cluster(object):

    def __init__(self, center):
        self.center = center
        self.data = []  # podaci koji pripadaju ovom klasteru

    def recalculate_center(self):
        # TODO 1: implementirati racunanje centra klastera
        # centar klastera se racuna kao prosecna vrednost svih podataka u klasteru
        new_center = [0 for i in range(len(self.center))]
        for d in self.data:
            for i in range(len(d)):
                new_center[i] += d[i]

        n = len(self.data)
        if n != 0:
            self.center = [x/n for x in new_center]


class KMeans(object):

    def __init__(self, n_clusters, max_iter):
        """
        :param n_clusters: broj grupa (klastera)
        :param max_iter: maksimalan broj iteracija algoritma
        :return: None
        """
        self.data = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.clusters = []
        self.prediction = []

    def fit(self, data, normalize=True):
        self.data = data  # lista N-dimenzionalnih podataka
        # TODO 4: normalizovati podatke pre primene k-means
        if normalize:
            self.data = self.normalize_data(self.data)

        # TODO 1: implementirati K-means algoritam za klasterizaciju podataka
        # kada algoritam zavrsi, u self.clusters treba da bude "n_clusters" klastera (tipa Cluster)
        dimensions = len(self.data[0])

        # napravimo N random tacaka i stavimo ih kao centar klastera
        for i in range(self.n_clusters):
            point = [random.random() for x in range(dimensions)]
            self.clusters.append(Cluster(point))

        # inicijalizujemo predikciju
        self.prediction = [None] * len(data)

        iter_no = 0
        not_moves = False
        while iter_no <= self.max_iter and (not not_moves):
            # ispraznimo podatke klastera
            for cluster in self.clusters:
                cluster.data = []

            count = 0
            for d in self.data:
                # index klastera kom pripada tacka
                cluster_index = self.predict(d)

                # dodamo tacku u klaster kako bi izracunali centar
                self.clusters[cluster_index].data.append(d)
                self.prediction[count] = cluster_index
                count += 1

            # TODO (domaci): prosiriti K-means da stane ako se u iteraciji centri klastera nisu pomerili
            # preracunavanje centra
            not_moves = True
            for cluster in self.clusters:
                old_center = copy.deepcopy(cluster.center)
                cluster.recalculate_center()

                not_moves = not_moves and (cluster.center == old_center)

            print("Iter no: " + str(iter_no))
            iter_no += 1

    def predict(self, datum):
        # TODO 1: implementirati odredjivanje kom klasteru odredjeni podatak pripada
        # podatak pripada onom klasteru cijem je centru najblizi (po euklidskoj udaljenosti)
        # kao rezultat vratiti indeks klastera kojem pripada
        min_distance = None
        cluster_index = None
        for index in range(len(self.clusters)):
            distance = self.euclidean_distance(datum, self.clusters[index].center)
            if min_distance is None or distance < min_distance:
                cluster_index = index
                min_distance = distance

        return cluster_index

    def euclidean_distance(self, x, y):
        # euklidsko rastojanje izmedju 2 tacke
        sq_sum = 0
        for xi, yi in zip(x, y):
            sq_sum += (yi - xi)**2

        return sq_sum ** 0.5

    def normalize_data(self, data):
        # mean-std normalizacija
        cols = len(data[0])

        for col in range(cols):
            column_data = []
            for row in data:
                column_data.append(row[col])

            mean = numpy.mean(column_data)
            std = numpy.std(column_data)

            for row in data:
                row[col] = (row[col] - mean) / std

        return data

    def sum_squared_error(self):
        # TODO 3: implementirati izracunavanje sume kvadratne greske
        # SSE (sum of squared error)
        # unutar svakog klastera sumirati kvadrate rastojanja izmedju podataka i centra klastera
        sse = 0
        for cluster in self.clusters:
            for d in cluster.data:
                sse += self.euclidean_distance(cluster.center, d)

        return sse**2
