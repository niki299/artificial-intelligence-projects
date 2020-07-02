
from read_data import read_file
from analysis import *
from KMeans import KMeans

if __name__ == "__main__":
    #plot_data()
    data = read_file('data/credit_card_data.csv')

    # correlation_check(data)

    # kmeans_data = data.values
    # using elbow method we pick number of clusters to be 7
    # sse_plot(kmeans_data)

    visualization_2d(data)