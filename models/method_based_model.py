from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import TomekLinks
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier

from models.model_base import model_base
from repositories.smells_repository.method_smells_repository import method_smells_repository
import os

# xgboost fix
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.1.0-posix-seh-rt_v5-rev2\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb


class method_based_model(model_base):
    def __init__(self, classifier=xgb.XGBClassifier(reg_alpha=0.2)):
        model_base.__init__(self)
        self.classifier = classifier
        self.method_based_smells = ["LongMethod", "FeatureEnvy"]
        self.smell_proportion = 0.06
        self.pu_adapter_enabled = True

    def get_classifier(self, smell):
        return self.classifier

    def get_dataset(self, smell):
        df = method_smells_repository().get_smells_dataset_from_projects(smell, self.projects_ids, self.dataset_ids)
        df["smell"] = smell
        return df

    def get_handled_smells(self):
        return self.method_based_smells

    def get_pipeline(self, smell):
        ppl = Pipeline([("scl", preprocessing.StandardScaler()),
                        #("ovs", SMOTETomek(ratio=self.get_ratio,smote=SMOTE(k_neighbors=5, ratio=self.get_ratio), tomek=TomekLinks(ratio=self.get_ratio))),
                        ("clf", self.get_puAdapter(smell))])
        return ppl