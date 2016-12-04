import numpy as np
from pylab import *
from sklearn import svm, datasets

# Creates fake income/age clusters for N people in k clusters
# Also it return training data to svm to work on!
def createClusteredData(N, k):
    pointsPerCluster = float(N)/k
    X = []
    y = []
    for i in range (k):
        incomeCentroid = np.random.uniform(20000.0, 200000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])
            y.append(i)
    X = np.array(X)
    y = np.array(y)
    return X, y


def support_vector_machines():
    (X, y) = createClusteredData(100, 5)
    C = 1.0

    # Using Linear kerenl for providig backbone to SVM
    clf = svm.SVC(kernel='linear', C=C).fit(X, y)
    xx, yy = np.meshgrid(np.arange(0, 250000, 10),
                     np.arange(10, 70, 0.5))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Plotting the svm classifying 100 people with their respective ages
    plt.figure(figsize=(8, 6))
    plt.title('Using Support Vector Machines\nFor 100 people\'s income with their ages')
    plt.xlabel('Income Scale')
    plt.ylabel('Age')
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c=y.astype(np.float))
    plt.show()

    # Print prediction model of 2 cluster found in which group
    print(' \n Prediction of 2 person found in which group'
            '\n Here Groups are namely: [0] ,[1] ,[2] ,[3] ,[4]')
    print('\n Person of income: 200000 & age: 40')
    print(clf.predict([[200000, 40]]))
    print('\n Person of income: 50000 & age: 65')
    print(clf.predict([[50000, 65]]))
    print('')
