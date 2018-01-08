from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import TomekLinks
from sklearn import preprocessing
import os
from sklearn.metrics import f1_score

# xgboost fix
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev2\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb

from models.model_base import model_base
from repositories.smells_repository.blob_repository import blob_repository


class class_metrics_model(model_base):
    def __init__(self, classifier=xgb.XGBClassifier(booster="dart", max_depth=5, reg_lambda=0.9, reg_alpha=0.15, subsample=0.8)):
        model_base.__init__(self)
        self.classifier = classifier
        self.class_metrics_smells = ["Blob"]
        self.samples_proportion = 0.4
        self.smell_proportion = 0.09
        self.pu_adapter_enabled = False

    def get_classifier(self, smell):
        return self.classifier

    def get_dataset(self, smell):
        return blob_repository().get_smells_dataset_from_projects(smell, self.projects_ids, self.dataset_ids)

    def get_handled_smells(self):
        return self.class_metrics_smells

    def get_pipeline(self, smell):
        ppl = Pipeline([("scl", preprocessing.StandardScaler()),
                        ("ovs", SMOTETomek(ratio=self.get_ratio,smote=SMOTE(k_neighbors=4, ratio=self.get_ratio), tomek=TomekLinks(ratio=self.get_ratio))),
                        ("clf", self.get_puAdapter(smell))])
        return ppl

