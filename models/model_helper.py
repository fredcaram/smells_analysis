from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

import numpy as np

from smells_dataset_handler.smells_dataset_handler import smells_dataset_handler

class model_helper:
    def __init__(self, classifier = tree.DecisionTreeClassifier()):
        self.classifier = classifier
        self.handled_smells = ["FeatureEnvy", "Blob", "LongMethod"]
        self.projects_ids = [108, 109]

    def get_balanced_train_test_split(self, X_data, y):
        X_resampled, y_resampled = SMOTEENN(random_state=42).fit_sample(X_data, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)
        return X_train, X_test, y_train, y_test

    def run_train_test_validation(self):
        projects = smells_dataset_handler().get_smells_dataset_from_projects(self.projects_ids)
        X_data = projects.iloc[:, 1:-5]

        for smell in self.handled_smells:
            y = projects[smell]
            X_train, X_test, y_train, y_test = self.get_balanced_train_test_split(X_data, y)
            trained_classifier = self.get_trained_model(X_train, y_train)
            print("Results for smell:{0}".format(smell))
            print("Score: {0}".format(trained_classifier.score(X_test, y_test)))
            y_pred = trained_classifier.predict(X_test)
            self.print_score(y_pred, y_test)

    def run_train_test_validation_with_cv(self):
        projects = smells_dataset_handler().get_smells_dataset_from_projects(self.projects_ids)
        X_data = projects.iloc[:, 1:-5]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for smell in self.handled_smells:
            y = projects[smell]
            X_resampled, y_resampled = SMOTEENN(random_state=42).fit_sample(X_data, y)
            scores = []
            print("Results for smell: {0}".format(smell))
            i = 1
            for train_index, test_index in kf.split(X_resampled, y_resampled):
                X_train = X_resampled[train_index]
                X_test = X_resampled[test_index]
                y_train = y_resampled[train_index]
                y_test = y_resampled[test_index]
                trained_classifier = self.get_trained_model(X_train, y_train)
                y_pred = trained_classifier.predict(X_test)
                scores.append(precision_recall_fscore_support(y_test, y_pred, average="micro")[0:3])
                print("Iteration {0} accuracy: {1}".format(i, trained_classifier.score(X_test, y_test)))
                i += 1

            print("Precision, Recall, F1 Score, Support:")
            print(np.mean(scores, axis=0))

    def get_trained_model(self, X_train, y_train):
        trained_classifier = self.classifier.fit(X_train, y_train)
        return trained_classifier

    def print_score(self, y_pred, y_test):
        print("Precision, Recall, F1 Score, Support:")
        print(precision_recall_fscore_support(y_test, y_pred, average="micro"))