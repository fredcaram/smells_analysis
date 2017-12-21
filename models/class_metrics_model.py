from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import TomekLinks
from sklearn import preprocessing
from sklearn.svm import OneClassSVM

from models.model_base import model_base
from repositories.smells_repository.blob_repository import blob_repository


class class_metrics_model(model_base):
    def __init__(self, classifier=OneClassSVM(kernel="linear")):
        model_base.__init__(self)
        self.classifier = classifier
        self.class_metrics_smells = ["Blob"]
        self.smell_proportion = 0.1

    def get_classifier(self):
        return self.classifier

    def get_dataset(self):
        return blob_repository().get_smells_dataset_from_projects(self.projects_ids, self.dataset_ids)

    def get_handled_smells(self):
        return self.class_metrics_smells

    def get_pipeline(self):
        ppl = Pipeline([("scl", preprocessing.StandardScaler()),
                        #("ovs", SMOTETomek(ratio=self.get_ratio,smote=SMOTE(k_neighbors=5, ratio=self.get_ratio), tomek=TomekLinks(ratio=self.get_ratio))),
                        ("clf", self.get_puAdapter())])
        return ppl

