from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

default_models = [
    LinearSVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier
]


class EnsenbleModelBuilder(ClassifierMixin):
    def __init__(self, models_array=default_models, voting='hard'):
        ClassifierMixin.__init__(self)
        self.models = models_array
        self.voting = voting

    def create_ensemble_model(self):
        return VotingClassifier(estimators=self.models, voting=self.voting)

