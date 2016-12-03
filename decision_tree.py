import pandas as pd
import pydotplus
from sklearn import tree
from PIL import Image
from sklearn.externals.six import StringIO
from sklearn.ensemble import RandomForestClassifier

def dTree():
    # Here getting test data set i.e. PastHires.csv
    input_file = "./data/PastHires.csv"
    df = pd.read_csv(input_file, header = 0)
    df.head()

    # Molding the csv to understandable dataframe
    d = {'Y': 1, 'N': 0}
    df['Hired'] = df['Hired'].map(d)
    df['Employed?'] = df['Employed?'].map(d)
    df['Top-tier school'] = df['Top-tier school'].map(d)
    df['Interned'] = df['Interned'].map(d)
    d = {'UG': 0, 'PG': 1, 'PhD': 2}
    df['Level of Education'] = df['Level of Education'].map(d)
    df.head()

    # Picking out features to select a candidate i.e. experience , employed etc
    features = list(df.columns[:6])
    features

    # Tree parameterizing i.e. decision : hiring of candidate
    y = df["Hired"]
    X = df[features]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X,y)

    # Creating decision tree with training data i.e. PastHires.csv
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                             feature_names=features)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('decision_tree.png')
    Image.open("decision_tree.png").show()

    # Creating random 10 entries to make decision of 2 caandidates
    # 1 means hired and 0 means not hired
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X, y)
    #Predict employment of an employed 10-year veteran
    print(clf.predict([[10, 1, 4, 0, 0, 0]]))
    #...and an unemployed 10-year veteran
    print (clf.predict([[10, 0, 4, 0, 0, 0]]))
