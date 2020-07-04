import matplotlib.pyplot as plt
import seaborn as sb
from KMeans import KMeans
import pandas as pd
from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans

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

    # Do KMeans for PCA data   n_clusters(6 or 7)
    km = KMeans(n_clusters= 6, max_iter=200)
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

def attr_analysis(data):

    km = KMeans(n_clusters=6, max_iter=200)
    km.fit(data.values, True)

    for cluster in km.clusters:
        for i in range(len(cluster.data[0])):
            col = _column(cluster.data, i)
            ax = plt.subplot(3, 6, i + 1)
            ax.set_title(data.columns[i], {'fontsize' : 6})
            plt.boxplot(col)

        plt.show()

def _column(matrix, i):
    return [row[i] for row in matrix]

def exploratory_analysis(data):
    best_columns = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT"]
    # data with best col
    best_data = pd.DataFrame(data[best_columns])

    km = KMeans(n_clusters=6, max_iter=200)
    km.fit(best_data.values, True)

    best_data['cluster'] = km.prediction
    best_columns.append('cluster')

    sb.pairplot(best_data[best_columns], hue='cluster', x_vars=best_columns,
                 y_vars=best_columns,
                 height=5, aspect=1)

    sb.pairplot(best_data[best_columns], hue='cluster', x_vars=best_columns[0:4],
                y_vars='cluster',
                height=5, aspect=1)

    sb.pairplot(best_data[best_columns], hue='cluster', x_vars=best_columns[4:7],
                y_vars='cluster',
                height=5, aspect=1)

    plt.show()