import abc
import math

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from puAdapter import PUAdapter
from puScorer import PUScorer
import scipy.stats as st

from messages import error_messages


class model_base:
    def __init__(self, metaclass=abc.ABCMeta):
        self.smells_columns = ['Blob', 'FeatureEnvy', 'LongMethod', 'ShotgunSurgery',
                               "DivergentChange", "ParallelInheritance"]
        self.projects_ids = [49, 52, 54, 55, 56, 57, 60, 61, 63, 64, 70, 71, 72, 73, 77, 78, 79, 80, 81, 86, 108, 109,
                             110, 111, 112, 127, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126]
        self.dataset_ids = [1, 2]
        self.remove_from_train = ["instance", "smell", "project_id"]
        self.smell_proportion = 0.1
        self.samples_proportion = 0.5
        self.pu_adapter_enabled = False
        self.negative_class = 0


    @abc.abstractmethod
    def get_dataset(self, smell):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_metrics'))


    @abc.abstractmethod
    def get_classifier(self, smell):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_classifier'))

    @abc.abstractmethod
    def get_pipeline(self, smell):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_pipeline'))

    def get_puAdapter(self, smell):
        if self.pu_adapter_enabled:
            return PUAdapter(estimator=self.get_classifier(smell), hold_out_ratio=self.smell_proportion)
        return self.get_classifier(smell)

    def get_ratio(self, y):
        non_smell_number = np.sum(y==0)
        return {0: non_smell_number, 1: math.ceil((non_smell_number+ 1) * self.samples_proportion)}

    def get_optimization_metrics(self):
        return {
            "clf__kernel": ["linear", "poly", "rbf"],
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

    # def get_balanced_data(self, X_data, y):
    #     print("Samples before balancing:")
    #     self.get_smells_proportion(y)
    #     ratio = {0: np.sum(y == 0), 1: math.ceil(np.sum(y == 0) * self.smell_proportion)}
    #     balancer = SMOTETomek(ratio=ratio, smote=SMOTE(k_neighbors=3, ratio=ratio), tomek=TomekLinks(ratio=ratio))
    #     X_resampled, y_resampled = balancer.fit_sample(X_data, y)
    #     print("Samples after balancing:")
    #     self.get_smells_proportion(y_resampled)
    #     return X_resampled, y_resampled



    def get_train_test_split(self, X_data, y):
        X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2)
        return X_train, X_test, y_train, y_test


    def train_model(self, X_train, y_train, smell):
        clf = self.get_pipeline(smell)
        trained_clf = clf.fit(X_train, y_train)
        return trained_clf


    def get_prediction(self, trained_classifier, X_test):
        return trained_classifier.predict(X_test)


    def get_X_features(self, projects):
        X_data = projects.drop(
            [col for col in self.remove_from_train + self.get_handled_smells() if col in projects.columns.values], axis=1, errors="ignore")
        return X_data


    def get_y_feature(self, projects, smell):
        y = np.array(projects[smell])
        if self.negative_class != 0:
            y[y==0] = -1
        return y

    def get_smells_stats(self, projects, smell, interval):
        smells_by_project_id = projects.groupby("project_id").aggregate({smell: "sum"})
        projects_means = smells_by_project_id[smell] / len(smells_by_project_id)
        total_mean = np.mean(projects_means)
        ci_lb, ci_ub = st.t.interval(0.95, len(projects_means) - 1, loc=np.mean(total_mean), scale=st.sem(projects_means))
        return {"mean": total_mean, "ci_lb": ci_lb, "ci_ub": ci_ub}

    def run_train_test_validation(self):
        for smell in self.get_handled_smells():
            projects = self.get_dataset(smell)
            X_data = self.get_X_features(projects)

            if not smell in projects.columns.values:
                continue

            y = self.get_y_feature(projects, smell)

            if len(np.unique(y)) < 2:
                continue

            print("Training Smell:{0}".format(smell))
            X_train, X_test, y_train, y_test = self.get_train_test_split(X_data, y)
            trained_classifier = self.train_model(X_train, y_train, smell)
            print("Results for smell:{0}".format(smell))
            #self.get_score(trained_classifier, X_test, y_test)
            y_pred = self.get_prediction(trained_classifier, X_test)
            self.print_score(y_pred, y_test)
            self.get_pu_score(y_pred, y_test, True)

    def run_cv_validation(self):
        for smell in self.get_handled_smells():
            projects = self.get_dataset(smell)
            X_data = self.get_X_features(projects)
            smell_stats = self.get_smells_stats(projects, smell, 0.95)

            if not smell in projects.columns.values:
                continue

            y = self.get_y_feature(projects, smell)

            if len(np.unique(y)) < 2:
                continue

            print("Non-Smells: {0}".format(np.sum(y == 0)))
            print("Smells: {0}".format(np.sum(y == 1)))

            scores = []
            pu_scores = []
            for train_index, test_index in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_data, y):
                print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X_data.iloc[train_index,:], X_data.iloc[test_index,:]
                y_train, y_test = y[train_index], y[test_index]

                print("Training Smell:{0}".format(smell))
                trained_classifier = self.train_model(X_train, y_train, smell)
                print("Results for smell:{0}".format(smell))
                #self.get_score(trained_classifier, X_test, y_test)
                #self.get_classifier().set_params(nu=(np.sum(y==1)/len(y)))
                y_pred = self.get_prediction(trained_classifier, X_test)
                scores.append(self.print_score(y_pred, y_test))
                pu_scores.append(self.get_pu_score(y_pred, y_test, smell_stats, True))

            print("Precision, Recall, F1 Score:")
            scores = np.delete(scores, -1, axis=1)
            print(np.mean(scores, axis=0))

            print("PU scores: Precision, Recall, F1 Score:")
            print(np.mean(pu_scores, axis=0))


    def run_random_search_cv(self):
        for smell in self.get_handled_smells():
            projects = self.get_dataset(smell)
            X_data = self.get_X_features(projects)

            if not smell in projects.columns.values:
                continue

            y = self.get_y_feature(projects, smell)

            if len(np.unique(y)) < 2:
                continue

            print("Non-Smells: {0}".format(np.sum(y == 0)))
            print("Smells: {0}".format(np.sum(y == 1)))

            X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2)

            print("Results for smell: {0}".format(smell))
            clf = self.get_pipeline(smell)

            rcv = RandomizedSearchCV(clf, param_distributions=self.get_optimization_metrics(), scoring="f1", n_iter=3, cv=3)
            rcv.fit(X_train, y_train)
            y_pred = rcv.predict(X_test)

            # self.get_score(cclf, X_test, y_test)
            self.print_score(y_pred, y_test)
            self.get_pu_score(y_pred, y_test, True)
            print("Best params:")
            print(rcv.best_params_)

    def run_balanced_classifier_cv(self):
        for smell in self.get_handled_smells():
            projects = self.get_dataset(smell)
            X_data = self.get_X_features(projects)

            if not smell in projects.columns.values:
                continue

            y = self.get_y_feature(projects, smell)

            if len(np.unique(y)) < 2:
                continue

            print("Non-Smells: {0}".format(np.sum(y == 0)))
            print("Smells: {0}".format(np.sum(y == 1)))

            X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2)

            print("Results for smell: {0}".format(smell))
            clf = self.get_pipeline(smell)

            cclf = CalibratedClassifierCV(clf, cv=8)
            cclf.fit(X_train, y_train)
            y_pred = cclf.predict(X_test)

            #self.get_score(cclf, X_test, y_test)
            self.print_score(y_pred, y_test)
            self.get_pu_score(y_pred, y_test, True)



    def print_score(self, y_pred, y_test):
        print("Precision, Recall, F1 Score, Support:")
        prec_rec_f = precision_recall_fscore_support(y_test, y_pred, average="binary")
        print(prec_rec_f)

        return prec_rec_f

    def get_pu_score(self, y_pred, y_test, smell_stats, print_score):
        pu_scorer = PUScorer(self.smell_proportion, y_test, np.ravel(y_pred), smell_stats)
        pu_prec = pu_scorer.get_precision()
        pu_rec = pu_scorer.get_recall()
        pu_f = pu_scorer.get_f_measure(pu_rec, pu_prec)
        if print_score:
            print("PU adjusted precision, recall and F1 score")
            print("{0}, {1}, {2}".format(pu_prec, pu_rec, pu_f))
        return pu_prec, pu_rec, pu_f


    def print_features(self, trained_classifier, X_features_columns):
        print("Relevant Features:")
        features = trained_classifier.feature_importances_
        df = pd.DataFrame(features)
        #df = pd.DataFrame(features, index=X_features_columns)
        print(df)
        return features