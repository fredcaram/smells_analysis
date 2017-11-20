import abc
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, cross_val_predict
from messages import error_messages


class model_base:
    def __init__(self, metaclass=abc.ABCMeta):
        self.smells_columns = ['Blob', 'FeatureEnvy', 'LongMethod', 'ShotgunSurgery',
                               "DivergentChange", "ParallelInheritance"]
        self.projects_ids = [49, 52, 54, 55, 56, 57, 60, 61, 63, 64, 70, 71, 72, 73, 77, 78, 79, 80, 81, 86]
        self.dataset_ids = [1]
        self.remove_from_train = ["instance"]


    @abc.abstractproperty
    def get_dataset(self):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_metrics'))


    @abc.abstractproperty
    def get_classifier(self):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_classifier'))


    @abc.abstractproperty
    def get_handled_smells(self):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_handled_smells'))

    def get_score(self, trained_classifier, X_test, y_test):
        print("Score: {0}".format(trained_classifier.score(X_test, y_test)))


    def get_balanced_data(self, X_data, y):
        X_resampled, y_resampled = SMOTEENN(random_state=42).fit_sample(X_data, y)
        return X_resampled, y_resampled


    def get_train_test_split(self, X_data, y):
        X_resampled, y_resampled = self.get_balanced_data(X_data, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)
        return X_train, X_test, y_train, y_test


    def train_model(self, X_train, y_train):
        trained_clf = self.get_classifier().fit(X_train, y_train)
        return trained_clf


    def get_prediction(self, trained_classifier, X_test):
        return trained_classifier.predict(X_test)


    def get_X_features(self, projects):
        X_data = projects.drop(
            [col for col in self.remove_from_train + self.get_handled_smells() if col in projects.columns.values], axis=1)
        return X_data


    def get_y_feature(self, projects, smell):
        y = projects[smell]
        return y


    def run_train_test_validation(self):
        projects = self.get_dataset()
        X_data = self.get_X_features(projects)

        for smell in self.get_handled_smells():
            if not smell in projects.columns.values:
                continue

            y = self.get_y_feature(projects, smell)

            if len(np.unique(y)) < 2:
                continue

            print("Training Smell:{0}".format(smell))
            X_train, X_test, y_train, y_test = self.get_train_test_split(X_data, y)
            trained_classifier = self.train_model(X_train, y_train)
            print("Results for smell:{0}".format(smell))
            self.get_score(trained_classifier, X_test, y_test)
            self.print_features(trained_classifier, X_data.columns.values)
            y_pred = self.get_prediction(trained_classifier, X_test)
            self.print_score(y_pred, y_test)

    def run_train_test_validation_with_cv(self):
        projects = self.get_dataset()
        X_data = self.get_X_features(projects)

        for smell in self.get_handled_smells():
            if not smell in projects.columns.values:
                continue

            y = self.get_y_feature(projects, smell)

            if len(np.unique(y)) < 2:
                continue

            X_resampled, y_resampled = self.get_balanced_data(X_data, y)
            print("Results for smell: {0}".format(smell))
            clf = self.get_classifier()

            score = cross_val_score(clf, X_resampled, y_resampled, cv=5)
            print("Score: {0}".format(np.mean(score)))
            y_pred = cross_val_predict(clf, X_resampled, y_resampled, cv=5)

            self.print_score(y_pred, y_resampled)


    def print_score(self, y_pred, y_test):
        print("Precision, Recall, F1 Score, Support:")
        prec_rec_f = precision_recall_fscore_support(y_test, y_pred, average="binary")
        print(prec_rec_f)


    def print_features(self, trained_classifier, X_features_columns):
        print("Relevant Features:")
        features = trained_classifier.feature_importances_
        df = pd.DataFrame(features, index=X_features_columns)
        print(df)