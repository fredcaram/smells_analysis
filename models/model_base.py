import abc
import math

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from messages import error_messages


class model_base:
    def __init__(self, metaclass=abc.ABCMeta):
        self.smells_columns = ['Blob', 'FeatureEnvy', 'LongMethod', 'ShotgunSurgery',
                               "DivergentChange", "ParallelInheritance"]
        self.projects_ids = [49, 52, 54, 55, 56, 57, 60, 61, 63, 64, 70, 71, 72, 73, 77, 78, 79, 80, 81, 86, 108, 109,
                             110, 111, 112, 127, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126]
        self.dataset_ids = [1, 2]
        self.remove_from_train = ["instance"]
        self.smell_proportion = 0.08


    @abc.abstractproperty
    def get_dataset(self):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_metrics'))


    @abc.abstractproperty
    def get_classifier(self):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_classifier'))

    @abc.abstractproperty
    def get_pipeline(self):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_pipeline'))

    def get_ratio(self, y):
        non_smell_number = np.sum(y==0)
        return {0: non_smell_number, 1: math.ceil(non_smell_number * self.smell_proportion)}

    def get_optimization_metrics(self, n_features):
        return {
            "ovs__kind": ["regular", "borderline1", "borderline2"],
            #"clf__n_estimators ": sp_randint(40, 100),
            #"clf__learning_rate ": sp_randint(.95, 1.05)
            #"clf__max_depth": sp_randint(1, 8),
            #"clf__max_features": sp_randint(1, n_features),
            #"clf__min_samples_split": sp_randint(2, 11),
            #"clf__min_samples_leaf": sp_randint(1, 11),
            #"clf__criterion": ["gini", "entropy"]
        }


    @abc.abstractproperty
    def get_handled_smells(self):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_handled_smells'))

    def get_score(self, trained_classifier, X_test, y_test):
        print("Score: {0}".format(trained_classifier.score(X_test, y_test)))

    def get_smells_proportion(self, y):
        nsmells = np.sum(y == 1)
        print("Non-Smells: {0}".format(np.sum(y == 0)))
        print("Smells: {0}".format(nsmells))
        print("Smells Proportion: {0}".format(nsmells / len(y)))

    def get_balanced_data(self, X_data, y):
        print("Samples before balancing:")
        self.get_smells_proportion(y)
        ratio = {0: np.sum(y == 0), 1: math.ceil(np.sum(y == 0) * self.smell_proportion)}
        balancer = SMOTETomek(ratio=ratio, smote=SMOTE(k_neighbors=3, ratio=ratio), tomek=TomekLinks(ratio=ratio))
        X_resampled, y_resampled = balancer.fit_sample(X_data, y)
        print("Samples after balancing:")
        self.get_smells_proportion(y_resampled)
        return X_resampled, y_resampled


    def get_train_test_split(self, X_data, y):
        X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2)
        #X_train_resampled, y_train_resampled = self.get_balanced_data(X_train, y_train)
        return X_train, X_test, y_train, y_test


    def train_model(self, X_train, y_train):
        clf = self.get_pipeline(np.sum(y_train == 0))
        trained_clf = clf.fit(X_train, y_train)
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
            #self.print_features(trained_classifier, X_data.columns.values)
            y_pred = self.get_prediction(trained_classifier, X_test)
            self.print_score(y_pred, y_test)

    def run_cv_validation(self):
        projects = self.get_dataset()
        X_data = self.get_X_features(projects)

        for smell in self.get_handled_smells():
            if not smell in projects.columns.values:
                continue

            y = self.get_y_feature(projects, smell)

            if len(np.unique(y)) < 2:
                continue

            features = []
            scores = []
            for train_index, test_index in KFold(n_splits=5, shuffle=True, random_state=42).split(X_data, y):
                print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X_data.iloc[train_index,:], X_data.iloc[test_index,:]
                y_train, y_test = y[train_index], y[test_index]

                X_train_resampled, y_train_resampled = self.get_balanced_data(X_train, y_train)
                print("Training Smell:{0}".format(smell))
                trained_classifier = self.train_model(X_train_resampled, y_train_resampled)
                print("Results for smell:{0}".format(smell))
                self.get_score(trained_classifier, X_test, y_test)
                features.append(self.print_features(trained_classifier, X_data.columns.values))
                y_pred = self.get_prediction(trained_classifier, X_test)
                scores.append(self.print_score(y_pred, y_test))

            print("Precision, Recall, F1 Score, Support:")
            scores = np.delete(scores, -1, axis=1)
            print(np.mean(scores, axis=0))

            features_df = pd.DataFrame(np.mean(features, axis=0), index=X_data.columns.values)
            print("Relevant Features:")
            print(features_df)


    def run_balanced_classifier_cv(self):
        projects = self.get_dataset()
        X_data = self.get_X_features(projects)

        for smell in self.get_handled_smells():
            if not smell in projects.columns.values:
                continue

            y = self.get_y_feature(projects, smell)

            if len(np.unique(y)) < 2:
                continue

            print("Non-Smells: {0}".format(np.sum(y == 0)))
            print("Smells: {0}".format(np.sum(y == 1)))

            X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2)

            #X_resampled, y_resampled = self.get_balanced_data(X_data, y)
            print("Results for smell: {0}".format(smell))
            clf = self.get_pipeline()

            #score = cross_val_score(clf, X_resampled, y_resampled, cv=10)
            #print("Score: {0}".format(np.mean(score)))
            #random_search = RandomizedSearchCV(clf, param_distributions=self.get_optimization_metrics(n_features=X_data.shape[1]),
            #                                   n_iter=10, scoring='f1', cv=3)
            #random_search.fit(X_train, y_train)
            cclf = CalibratedClassifierCV(clf, cv=8)
            cclf.fit(X_train, y_train)
            y_pred = cclf.predict(X_test)

            self.get_score(cclf, X_test, y_test)
            self.print_score(y_pred, y_test)


    def print_score(self, y_pred, y_test):
        print("Precision, Recall, F1 Score, Support:")
        prec_rec_f = precision_recall_fscore_support(y_test, y_pred, average="binary")
        print(prec_rec_f)
        return prec_rec_f


    def print_features(self, trained_classifier, X_features_columns):
        print("Relevant Features:")
        features = trained_classifier.feature_importances_
        df = pd.DataFrame(features)
        #df = pd.DataFrame(features, index=X_features_columns)
        print(df)
        return features