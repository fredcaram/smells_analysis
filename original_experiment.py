from arff2pandas import a2p
import pandas as pd
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_val_score, cross_val_predict, cross_validate, \
    GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import os
from sklearn.externals import joblib

from weka.classifiers import Classifier, Evaluation, GridSearch
from weka.core.converters import Loader
from weka.core.classes import Random

from rpy2.robjects import r, pandas2ri, numpy2ri

class OriginalExperiment:
    @staticmethod
    def read_arff(filename:str):
        with open(filename) as f:
            df = a2p.load(f)
            return df

    @staticmethod
    def get_clean_data(data):
        new_data = data[[col for col in data.columns.values
                         if col not in ["id", "project", "package", "complextype", "method"]]]
        #new_data = new_data.iloc[:, :-26]
        return new_data

    @staticmethod
    def get_x(data):
        new_data = data[[col for col in data.columns.values
                      if col not in ["id", "project", "package", "complextype", "method", "is_smell"]]]
        new_data = new_data.iloc[:, :-26]
        return new_data

    @staticmethod
    def get_y(data):
        return data["is_smell"].values == "true"

    @staticmethod
    def get_y_asframe(data):
        return data["is_smell"]

    @staticmethod
    def get_x_and_y_from_data(data):
        return OriginalExperiment.get_x(data), OriginalExperiment.get_y(data)

    @staticmethod
    def train_and_tune_r_tree(data, name):
        X_data, y_data = OriginalExperiment.get_x_and_y_from_data(data)
        pandas2ri.activate()
        r_x = pandas2ri.py2ri(X_data)
        r_y = numpy2ri.py2ri(y_data.astype(int))
        r.options(warn=1)
        r.source(os.path.join("rsrc", "original_experiment_replication.R"))
        seed = 42
        r_model = r.boosted_pruned_dtree_gridcv(r_x, r_y, name, seed)
        print(r_model)


    @staticmethod
    def tune_classifier(classifier, params_grid, data):
        X_data, y_data = OriginalExperiment.get_x_and_y_from_data(data)
        clf = GridSearchCV(classifier, param_grid=params_grid,
                           cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=42),
                           scoring="f1_micro", n_jobs=2, return_train_score=False)
        clf.fit(X_data, y_data)

        OriginalExperiment.print_feature_importances(X_data.columns.values, clf.best_estimator_)

        return clf

    @staticmethod
    def print_feature_importances(columns, estimator):
        features = {}
        for i, feature in enumerate(columns):
            features[feature] = estimator.feature_importances_[i]
        print(features)

    @staticmethod
    def model_cross_validate(classifier, data):
        X_data, y_data = OriginalExperiment.get_x_and_y_from_data(data)
        scores = cross_validate(classifier, X_data, y_data,
                                cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42),
                                scoring=["precision", "recall", "f1_micro"], n_jobs=2)
        print(np.mean(scores["test_precision"]))
        print(np.mean(scores["test_recall"]))
        print(np.mean(scores["test_f1_micro"]))
        #OriginalExperiment.print_score(y_pred, y_data, True)
        print(classifier.get_params())

    @staticmethod
    def print_score(y_pred, y_test, print_score):
        prec_rec_f = precision_recall_fscore_support(y_test, y_pred, average="micro")
        if print_score:
            print("Precision, Recall, F1 Score, Support:")
            print(prec_rec_f)

        return prec_rec_f

    @staticmethod
    def load_model(model_name):
        filename = os.path.join(".", "original_experiment_models", "{0}.pkl".format(model_name))
        if not os.path.exists(filename):
            return None

        return joblib.load(filename)

    @staticmethod
    def save_model(model_name, model):
        filename = os.path.join(".", "original_experiment_models", "{0}.pkl".format(model_name))

        joblib.dump(model, filename)