import abc

import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from messages import error_messages


class model_base:
    def __init__(self, metaclass=abc.ABCMeta):
        self.smells_columns = ['Blob', 'FeatureEnvy', 'LongMethod', 'ShotgunSurgery',
                               "DivergentChange", "ParallelInheritance"]
        self.projects_ids = [54, 55, 64]
        self.remove_from_train = ["instance"]

    @abc.abstractproperty
    def get_dataset(self):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_metrics'))

    @abc.abstractproperty
    def get_handled_smells(self):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_handled_smells'))

    @abc.abstractmethod
    def train_model(self, X_train, y_train):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('train_model'))

    @abc.abstractmethod
    def get_balanced_data(self, X_data, y):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_balanced_data'))

    @abc.abstractmethod
    def get_train_test_split(self, X_data, y):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_train_test_split'))

    @abc.abstractmethod
    def get_prediction(self, trained_classifier, X_test):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_prediction'))

    @abc.abstractmethod
    def get_X_features(self, projects):
        X_data = projects.drop(
            [col for col in self.remove_from_train + self.smells_columns if col in projects.columns.values], axis=1)
        return X_data

    @abc.abstractmethod
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
            print("Score: {0}".format(trained_classifier.score(X_test, y_test)))
            y_pred = self.get_prediction(X_test)
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
            scores = self.run_cross_val(X_resampled, y_resampled)
            print("Precision, Recall, F1 Score, Support:")
            print(np.mean(scores, axis=0))

    def run_cross_val(self, X_resampled, y_resampled):
        scores = []
        i = 1
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(X_resampled, y_resampled):
            X_train = X_resampled[train_index]
            X_test = X_resampled[test_index]
            y_train = y_resampled[train_index]
            y_test = y_resampled[test_index]
            trained_classifier = self.train_model(X_train, y_train)
            y_pred = self.get_prediction(X_test)
            scores.append(precision_recall_fscore_support(y_test, y_pred, average="micro")[0:3])
            print("Iteration {0} accuracy: {1}".format(i, trained_classifier.score(X_test, y_test)))
            i += 1
        return scores

    def print_score(self, y_pred, y_test):
        print("Precision, Recall, F1 Score, Support:")
        print(precision_recall_fscore_support(y_test, y_pred, average="micro"))