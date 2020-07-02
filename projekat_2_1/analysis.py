import matplotlib.pyplot as plt
import seaborn as sb
from KMeans import KMeans
from sklearn.decomposition import PCA

def sse_plot(X, start = 2, stop = 20):
    inertia = []
    for x in range(start,stop):
        print("====ITERATION:", x)
        km = KMeans(n_clusters = x, max_iter=1000)
        km.fit(X, True)
        inertia.append(km.sum_squared_error())
    plt.figure(figsize = (12,6))
    plt.plot(range(start,stop), inertia, marker = 'o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.title('Inertia plot with K')
    plt.xticks(list(range(start, stop)))
    plt.show()

def correlation_check(data):
    plt.figure(figsize=(12, 12))
    sb.heatmap(data.corr(), annot=True, cmap='coolwarm',
                xticklabels=data.columns,
                yticklabels=data.columns)

    plt.show()

def visualization_2d(data):

    # reduce dimesions of dataset based on data variance (PCA)
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)

    # Do KMeans for PCA data
    km = KMeans(n_clusters= 7, max_iter=200)
    km.fit(pca_data, True)

    colors = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'gray']
    for i in range(len(km.clusters)):
        pc1 = []
        pc2 = []
        for row in km.clusters[i].data:
            pc1.append(row[0])
            pc2.append(row[1])
        plt.scatter(pc1, pc2, c=colors[i], label='cluster ' + str(i))

    plt.show()

