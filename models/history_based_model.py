from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import TomekLinks
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

from models.dnn_models import simple_dnn
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from models.model_base import model_base
from repositories.smells_repository.relationships_smells_repository import relationship_smells_repository

import os
# xgboost fix
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev2\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb

class history_based_model(model_base):
    def __init__(self, classifier=RandomForestClassifier()):
        self.classifier = classifier

        model_base.__init__(self)
        self.history_based_smells = ['ShotgunSurgery', "DivergentChange"]#, "ParallelInheritance"
        self.smell_proportion = 0.0085
        self.samples_proportion = 0.5

    def get_classifier(self, smell):
        if type(self.classifier) is dict:
            return self.classifier[smell]

        return self.classifier

    def get_dataset(self, smell):
        df = relationship_smells_repository().get_smells_dataset_from_projects(smell, self.projects_ids, self.dataset_ids)
        return df

    def get_handled_smells(self):
        return self.history_based_smells


class divergent_change_model(history_based_model):
    def __init__(self, classifier=xgb.XGBClassifier()):
        self.classifier = classifier

        history_based_model.__init__(self)
        self.history_based_smells = ["DivergentChange"]#, "ParallelInheritance"
        self.smell_proportion = 0.0015
        self.samples_proportion = 0.4
        self.pu_adapter_enabled = True


    def get_pipeline(self, smell):
        return Pipeline([("scl", preprocessing.StandardScaler()),
                            ("ovs",
                             SMOTETomek(ratio=self.get_ratio, smote=SMOTE(k_neighbors=2, ratio=self.get_ratio),
                                       tomek=TomekLinks(ratio=self.get_ratio))),
                            ("clf", self.get_puAdapter(smell))])

class shotgun_surgery_model(history_based_model):
    def __init__(self, classifier=RandomForestClassifier()):
        self.classifier = classifier

        history_based_model.__init__(self)
        self.history_based_smells = ["ShotgunSurgery"]#, "ParallelInheritance"
        self.smell_proportion = 0.002
        self.samples_proportion = 0.5

    def get_pipeline(self, smell):
        return Pipeline([("scl", preprocessing.StandardScaler()),
                            ("ovs",
                             SMOTETomek(ratio=self.get_ratio, smote=SMOTE(k_neighbors=3, ratio=self.get_ratio),
                                        tomek=TomekLinks(ratio=self.get_ratio))),
                            ("clf", self.get_puAdapter(smell))])
