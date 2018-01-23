from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier


default_models = [
    ("knb", KNeighborsClassifier()),
    ("gnb", GaussianNB()),
    ("dtc", DecisionTreeClassifier()),
    ("rfc", RandomForestClassifier()),
    ("log", LogisticRegression())
]

default_weights = [1, 1, 1, 1, 1]

class EnsenbleModelBuilder():
    def __init__(self, models_array=default_models, voting='soft', weights=default_weights):
        self.models = models_array
        self.voting = voting
        self.weights = weights

    def create_ensemble_model(self):
        return VotingClassifier(estimators=self.models, voting=self.voting, weights=self.weights)




