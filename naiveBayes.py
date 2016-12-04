import os
import io
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Read files from path recursively for training!
def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)
            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message

# Creating data frame : spam / ham
def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)
    return DataFrame(rows, index=index)

def nBayes():
    # Creating pandas dataframe consisting : message & its Class : Spam/ham
    data = DataFrame({'message': [], 'class': []})

    data = data.append(dataFrameFromDirectory('./data/emails/spam', 'spam'))
    data = data.append(dataFrameFromDirectory('./data/emails/ham', 'ham'))

    data.head()

    # Converting Message into its list of words
    vectorizer = CountVectorizer()
    counts = vectorizer.fit_transform(data['message'].values)

    # Using MultinomialNB classifier
    classifier = MultinomialNB()
    targets = data['class'].values
    classifier.fit(counts, targets)

    # Testing out email to get result
    examples = ['Demonitization Rocks!!!', "Free Money Now!!"]
    example_counts = vectorizer.transform(examples)
    predictions = classifier.predict(example_counts)
    print(' Emails: Demonitization Rocks!!! ,Free Money Now!! \n Results:')
    print(predictions)
    print('')
