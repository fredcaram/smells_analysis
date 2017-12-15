from sklearn import tree

from models.model_base import model_base
from repositories.smells_repository.blob_repository import blob_repository


class class_metrics_model(model_base):
    def __init__(self, classifier=tree.DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, max_features="auto")):
        model_base.__init__(self)
        self.classifier = classifier
        self.class_metrics_smells = ["Blob"]

    def get_classifier(self):
        return self.classifier

    def get_dataset(self):
        return blob_repository().get_smells_dataset_from_projects(self.projects_ids, self.dataset_ids)

    def get_handled_smells(self):
        return self.class_metrics_smells

