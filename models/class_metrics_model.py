from sklearn import tree

from models.model_base import model_base
from smells_dataset_handler.metric_reloaded_class_smells_dataset_handler import metric_reloaded_class_smells_dataset_handler


class class_metrics_model(model_base):
    def __init__(self, classifier=tree.DecisionTreeClassifier()):
        model_base.__init__(self)
        self.classifier = classifier
        self.class_metrics_smells = ["Blob"]

    def get_classifier(self):
        return self.classifier

    def get_dataset(self):
        return metric_reloaded_class_smells_dataset_handler().get_smells_dataset_from_projects(self.projects_ids)

    def get_handled_smells(self):
        return self.class_metrics_smells

