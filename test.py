import pandas as pd
import time
from sklearn.model_selection import train_test_split
from project import my_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def test(data):
    y = data["fraudulent"]
    X = data.drop(['fraudulent'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Train model
    clf = my_model()
    scores = clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return predictions, y_test, scores


if __name__ == "__main__":
    start = time.time()
    # Load data
    data = pd.read_csv("../data/job_train.csv")
    predictions, y_test, scores = test(data)
    print("\n f1_score", scores)
    runtime = (time.time() - start) / 60.0
    print('\n Training Time: ')
    print(runtime)


