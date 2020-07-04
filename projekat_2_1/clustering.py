
from read_data import read_file
from analysis import *

if __name__ == "__main__":
    #plot_data()
    data = read_file('data/credit_card_data.csv')
    ''' 
    show matrix of correlation
    '''
    #correlation_check(data)

    # kmeans_data = data.values
    # using elbow method we pick number of clusters to be 6
    # sse_plot(kmeans_data)

    # attr_analysis(data)

    # visualization_2d(data)

    # exploratory_analysis(data)

    treeClassification(data)