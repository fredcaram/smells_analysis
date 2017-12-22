from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import TomekLinks
from sklearn import preprocessing
from sklearn.svm import SVC

from models.model_base import model_base
from repositories.smells_repository.relationships_smells_repository import relationship_smells_repository


class history_based_model(model_base):
    def __init__(self, classifier=None):
        if classifier is None:
            self.classifier = {"ShotgunSurgery": SVC(kernel="rbf"), "DivergentChange": SVC(kernel="sigmoid")}
        else:
            self.classifier = classifier

        model_base.__init__(self)
        self.history_based_smells = ['ShotgunSurgery', "DivergentChange"]#, "ParallelInheritance"
        self.smell_proportion = 0.009

    def get_classifier(self, smell):
        if type(self.classifier) is dict:
            return self.classifier[smell]

        return self.classifier

    def get_dataset(self):
        df = relationship_smells_repository().get_smells_dataset_from_projects(self.projects_ids, self.dataset_ids)
        return df

    def get_handled_smells(self):
        return self.history_based_smells

    def get_pipeline(self, smell):
        if smell == "ShotgunSurgery":
            ppl = Pipeline([("scl", preprocessing.StandardScaler()),
                            ("ovs", SMOTETomek(ratio=self.get_ratio,smote=SMOTE(k_neighbors=3, ratio=self.get_ratio), tomek=TomekLinks(ratio=self.get_ratio))),
                            ("clf", self.get_puAdapter(smell))])
        else:
            ppl = Pipeline([("scl", preprocessing.StandardScaler()),
                            ("ovs", SMOTETomek(ratio=self.get_ratio, smote=SMOTE(k_neighbors=2, ratio=self.get_ratio),
                                               tomek=TomekLinks(ratio=self.get_ratio))),
                            ("clf", self.get_puAdapter(smell))])
        return ppl