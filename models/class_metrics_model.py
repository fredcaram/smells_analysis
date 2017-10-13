from sklearn import tree
from sklearn.model_selection import train_test_split
from orangecontrib.associate.fpgrowth import association_rules, frequent_itemsets
from imblearn.combine import SMOTEENN

from models.model_base import model_base
from smells_dataset_handler.metric_reloaded_class_smells_dataset_handler import metric_reloaded_class_smells_dataset_handler, history_based_method_smells_dataset_handler


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

    def train_model(self, X_train, y_train):
        trained_classifier = self.get_classifier().fit(X_train, y_train)
        return trained_classifier

    def get_balanced_data(self, X_data, y):
        X_resampled, y_resampled = SMOTEENN(random_state=42).fit_sample(X_data, y)
        return X_resampled, y_resampled

    def get_prediction(self, trained_classifier, X_test):
        return trained_classifier.predict(X_test)

    def get_train_test_split(self, X_data, y):
        X_resampled, y_resampled = self.get_balanced_data(X_data, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)
        return X_train, X_test, y_train, y_test

class history_based_model(model_base):
    def __init__(self):
        model_base.__init__(self)
        self.class_metrics_smells = ["Blob"]

    def get_dataset(self):
        return history_based_method_smells_dataset_handler().get_smells_dataset_from_projects(self.projects_ids)

    def get_handled_smells(self):
        return self.class_metrics_smells

    def train_model(self, X_train, y_train):
        trained_classifier = association_rules(X_train, y_train)
        return trained_classifier

    def get_X_features(self, projects):
        X_data = frequent_itemsets(projects["instance"].values, 0.4)
        return X_data

    def get_y_feature(self, projects, smell):
        y = projects[smell]
        return y