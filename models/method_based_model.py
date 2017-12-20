from sklearn.svm import LinearSVC

from models.model_base import model_base
from repositories.smells_repository.method_smells_repository import method_smells_repository


class method_based_model(model_base):
    def __init__(self, classifier=LinearSVC()):
        model_base.__init__(self)
        self.classifier = classifier
        self.method_based_smells = ["LongMethod", "FeatureEnvy"]

    def get_classifier(self):
        return self.classifier

    def get_dataset(self):
        return method_smells_repository().get_smells_dataset_from_projects(self.projects_ids, self.dataset_ids)

    def get_handled_smells(self):
        return self.method_based_smells