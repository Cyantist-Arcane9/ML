from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from numpy import random, array, float

# Patching up fake income/age clusters for N people in k clusters
def createClusteredData(N, k):
    random.seed(10)
    pointsPerCluster = float(N)/k
    X = []
    for i in range (k):
        incomeCentroid = random.uniform(20000.0, 200000.0)
        ageCentroid = random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([random.normal(incomeCentroid, 10000.0), random.normal(ageCentroid, 2.0)])
    X = array(X)
    return X

# K-Means Clustering Method to make data more relatable to each other
def kMeans():
    data = createClusteredData(100, 5)

    # Using sklearn KMeans module
    model = KMeans(n_clusters=5)

    # Scaling the data to normalize / flattening it!
    model = model.fit(scale(data))

    # Visualize the scatter plot
    plt.figure(figsize=(8, 6))
    plt.title('Using K-Means Clustering\nFor 100 people\'s income with their ages')
    plt.xlabel('Income Scale')
    plt.ylabel('Age')
    plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(float))
    plt.show()
    print('\n Basically 5 clusters to iterate data in grouping')
