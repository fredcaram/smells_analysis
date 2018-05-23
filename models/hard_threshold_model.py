from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.pipeline import Pipeline
from sklearn.metrics import recall_score
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
from models.history_based_model import history_based_model
from repositories.smells_repository.inheritance_smells_repository import inheritance_smells_repository


DefaultThresholds = {"support": 0.008, "confidence": 0.7}

class HardThresholClassifierBase(BaseEstimator, ClassifierMixin):
    def __init__(self, thresholds=DefaultThresholds):
        self.thresholds = thresholds

    def fit(self, X, y=None):
        return

    def predict(self, X, y=None):
        y_pred = np.ones((X.shape[0]))
        for k, threshold in self.thresholds.items():
            assert (k in X.columns.values), "Dataset doesn't contain {0}".format(k)
            y_pred = np.logical_and(y_pred, X[k] > threshold)

        return y_pred

class DivergentChangeHardThresholdClassifier(HardThresholClassifierBase):
    def __init__(self, thresholds=DefaultThresholds):
        super().__init__(thresholds)
        self.thresholds = thresholds

    def fit(self, X, y=None):
        return

    def predict(self, X, y=None):
        y_pred = super().predict(X, y)
        #y_pred = np.logical_and(y_pred, X["cardinality"] >= 3)

        return y_pred


class DivergentChangeWithHardThresholdModel(history_based_model):
    def __init__(self):
        classifier = DivergentChangeHardThresholdClassifier()
        history_based_model.__init__(self, classifier)
        self.classifier = classifier
        self.history_based_smells = ["DivergentChange"]#, "ParallelInheritance"
        self.pu_adapter_enabled = False
        self.use_smote_tomek = False

    def get_pipeline(self, smell):
        pipeline_steps = []
        pipeline_steps.append(("clf", self.get_puAdapter(smell)))

        return Pipeline(pipeline_steps)


class ShotgunSurgeryHardThresholdClassifier(HardThresholClassifierBase):
    def __init__(self, thresholds=DefaultThresholds):
        super().__init__(thresholds)
        self.thresholds = thresholds

    def fit(self, X, y=None):
        return

    def predict(self, X, y=None):
        y_pred = super().predict(X, y)
        y_pred = np.logical_and(y_pred, X["cardinality"] >= 2)

        return y_pred


class ShotgunSurgeryWithHardThresholdModel(history_based_model):
    def __init__(self):
        classifier = ShotgunSurgeryHardThresholdClassifier()
        history_based_model.__init__(self, classifier)
        self.classifier = classifier
        self.history_based_smells = ["ShotgunSurgery"]#, "ParallelInheritance"
        self.pu_adapter_enabled = False
        self.use_smote_tomek = False

    def get_pipeline(self, smell):
        pipeline_steps = []
        pipeline_steps.append(("clf", self.get_puAdapter(smell)))

        return Pipeline(pipeline_steps)


class ParallelInheritanceHardThresholdClassifier(HardThresholClassifierBase):
    def __init__(self, thresholds=DefaultThresholds):
        super().__init__(thresholds)
        self.thresholds = thresholds

    def fit(self, X, y=None):
        return

    def predict(self, X, y=None):
        y_pred = super().predict(X, y)

        return y_pred


class ParallelInheritanceWithHardThresholdModel(history_based_model):
    def __init__(self):
        classifier = ParallelInheritanceHardThresholdClassifier()
        history_based_model.__init__(self, classifier)
        self.classifier = classifier
        self.history_based_smells = ["ParallelInheritance"]#, "ParallelInheritance"
        self.pu_adapter_enabled = False
        self.use_smote_tomek = False

    def get_classifier(self, smell):
        if type(self.classifier) is dict:
            return self.classifier[smell]

        return self.classifier

    def get_dataset(self, smell):
        df = inheritance_smells_repository().get_smells_dataset_from_projects(smell, self.projects_ids, self.dataset_ids)
        return df

    def get_handled_smells(self):
        return self.history_based_smells

    def get_pipeline(self, smell):
        pipeline_steps = []
        pipeline_steps.append(("clf", self.get_puAdapter(smell)))

        return Pipeline(pipeline_steps)