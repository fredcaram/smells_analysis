from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import TomekLinks
from sklearn import preprocessing
from models.ensemble_model import EnsenbleModelBuilder

from models.model_base import model_base
from repositories.smells_repository.method_smells_repository import method_smells_repository
import os

# xgboost fix
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev2\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb

import lightgbm as lgbm
import catboost as cat

class method_based_model(model_base):
    def __init__(self, classifier=xgb.XGBClassifier(reg_alpha=0.2)):
        model_base.__init__(self)
        self.classifier = classifier
        self.method_based_smells = ["LongMethod", "FeatureEnvy"]
        self.smell_weight = 0.06
        self.samples_proportion = 0.5
        self.pu_adapter_enabled = True

    def get_classifier(self, smell):
        return self.classifier

    def get_dataset(self, smell):
        df = method_smells_repository().get_smells_dataset_from_projects(smell, self.projects_ids, self.dataset_ids)
        df["smell"] = smell
        return df

    def get_handled_smells(self):
        return self.method_based_smells


class long_method_model(method_based_model):
    def __init__(self, classifier=xgb.XGBClassifier(reg_alpha=0.2)):
        method_based_model.__init__(self, classifier)
        self.classifier = classifier
        self.method_based_smells = ["LongMethod"]
        self.smell_weight = 0.018
        self.samples_proportion = 0.5
        self.pu_adapter_enabled = True
        self.use_smote_tomek = False

    def get_pipeline(self, smell):
        pipeline_steps = []
        if self.use_scaler:
            pipeline_steps.append(("scl", preprocessing.StandardScaler()))

        if self.use_smote_tomek:
            pipeline_steps.append(("ovs",
                             SMOTETomek(ratio=self.get_ratio, smote=SMOTE(k_neighbors=3, ratio=self.get_ratio),
                                       tomek=TomekLinks(ratio=self.get_ratio))),)
        pipeline_steps.append(("clf", self.get_puAdapter(smell)))

        return Pipeline(pipeline_steps)


class feature_envy_model(method_based_model):
    def __init__(self, classifier=EnsenbleModelBuilder(weights=[2.58204251, 0, 2.75587262, 0.69195661, 0]).create_ensemble_model()):
        method_based_model.__init__(self, classifier)
        self.classifier = classifier
        self.method_based_smells = ["FeatureEnvy"]
        self.smell_weight = 0.005
        self.samples_proportion = 0.5
        self.pu_adapter_enabled = False
        self.use_smote_tomek = True

    def get_pipeline(self, smell):
        pipeline_steps = []
        if self.use_scaler:
            pipeline_steps.append(("scl", preprocessing.StandardScaler()))

        if self.use_smote_tomek:
            pipeline_steps.append(("ovs",
                             SMOTETomek(ratio=self.get_ratio, smote=SMOTE(k_neighbors=3, ratio=self.get_ratio),
                                       tomek=TomekLinks(ratio=self.get_ratio))),)
        pipeline_steps.append(("clf", self.get_puAdapter(smell)))

        return Pipeline(pipeline_steps)