from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import TomekLinks
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from models.dnn_models import simple_dnn
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from models.model_base import model_base
from repositories.smells_repository.inheritance_smells_repository import inheritance_smells_repository

import os
# xgboost fix
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev2\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb

class parallel_inheritance_model(model_base):
    def __init__(self, classifier=LGBMClassifier()):
        self.classifier = classifier

        model_base.__init__(self)
        self.history_based_smells = ['ParallelInheritance']
        self.smell_proportion = 0.01
        self.samples_proportion = 0.5
        self.pu_adapter_enabled = True

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
        return Pipeline([("scl", preprocessing.StandardScaler()),
                            ("ovs",
                             SMOTETomek(ratio=self.get_ratio, smote=SMOTE(k_neighbors=3, ratio=self.get_ratio),
                                       tomek=TomekLinks(ratio=self.get_ratio))),
                            ("clf", self.get_puAdapter(smell))])
