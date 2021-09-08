import pandas as pd
import time
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=35)


class my_model():
    def fit(self, X, y):
        # do not exceed 29 mins
        data = pd.merge(X, y.to_frame(), how='right', left_index=True, right_index=True)
        data.interpolate(inplace=True)
        data.isnull().sum()
        data = data.fillna(1)
        del data['title']
        del data['location']
        del data['description']
        del data['requirements']
        data.rename(columns={0: 'fraudulent'}, inplace=True)

        # Class count
        count_class_0, count_class_1 = data.fraudulent.value_counts()

        # Divide by class
        df_class_0 = data[data['fraudulent'] == 0]
        df_class_1 = data[data['fraudulent'] == 1]
        # Over_Sampling
        # df_class_1_over = df_class_1.sample(df_class_0, replace=True)
        # data = pd.concat([df_class_0, df_class_1_over], axis=0)

        # Under_Sampling
        df_class_0_under = df_class_0.sample(count_class_1)
        data = pd.concat([df_class_0_under, df_class_1], axis=0)

        X = data.drop(['fraudulent'], axis=1)
        y = data["fraudulent"]

        classifier.fit(X, y)
        cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
        # scores=cross_val_score(classifier, X, y, cv=cv, scoring='f1')
        scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
        # scores=classifier.score(X,y)
        # print("\n f1_score" , scores)
        return scores

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        del X['title']
        del X['location']
        del X['description']
        del X['requirements']
        X = X.fillna(1)
        predictions = classifier.predict(X)
        return predictions
