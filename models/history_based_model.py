from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import TomekLinks
from sklearn import preprocessing
from sklearn.svm import LinearSVC

from models.model_base import model_base
from repositories.smells_repository.relationships_smells_repository import relationship_smells_repository


class history_based_model(model_base):
    def __init__(self, classifier=LinearSVC()):
        self.classifier = classifier
        model_base.__init__(self)
        self.history_based_smells = ['ShotgunSurgery', "DivergentChange"]#, "ParallelInheritance"
        self.smell_proportion = 0.01

    def get_classifier(self):
        return self.classifier

    def get_dataset(self):
        df = relationship_smells_repository().get_smells_dataset_from_projects(self.projects_ids, self.dataset_ids)
        return df

    def get_handled_smells(self):
        return self.history_based_smells

    def get_pipeline(self):
        ppl = Pipeline([("scl", preprocessing.StandardScaler()),
                        ("ovs", SMOTETomek(ratio=self.get_ratio,smote=SMOTE(k_neighbors=3, ratio=self.get_ratio), tomek=TomekLinks(ratio=self.get_ratio))),
                        ("clf", self.get_classifier())])
        return ppl