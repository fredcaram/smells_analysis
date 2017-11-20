
from sklearn.tree import DecisionTreeClassifier

from models.model_base import model_base
from repositories.smells_repository.relationships_smells_repository import relationship_smells_repository


class history_based_model(model_base):
    def __init__(self, classifier=DecisionTreeClassifier()):
        self.classifier = classifier
        model_base.__init__(self)
        self.history_based_smells = ['ShotgunSurgery', "DivergentChange"]#, "ParallelInheritance"

    def get_classifier(self):
        return self.classifier

    def get_dataset(self):
        df = relationship_smells_repository().get_smells_dataset_from_projects(self.projects_ids, self.dataset_ids)
        return df

    def get_handled_smells(self):
        return self.history_based_smells