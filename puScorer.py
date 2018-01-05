from sklearn.metrics import recall_score
import numpy as np
from pandas_ml import ConfusionMatrix


class PUScorer(object):
    def __init__(self, positive_proportion, y_true, y_pred):
        self._positive_proportion = positive_proportion
        self.conf_matrix = ConfusionMatrix(y_true, y_pred)

    def __get_true_positive__(self):
        original_tp = self.conf_matrix.TP
        recall = self.conf_matrix.TPR
        adjusted_tp = original_tp + self._positive_proportion * np.sum(self.conf_matrix.y_true() == 0) * recall
        return adjusted_tp

    def __get_false_positive__(self):
        original_fp = self.conf_matrix.FP
        fpr = self.conf_matrix.FPR
        adjusted_fp = original_fp + self._positive_proportion * np.sum(self.conf_matrix.y_true() == 0) * (1 - fpr)
        return adjusted_fp

    def __get_false_negative__(self):
        original_fn = self.conf_matrix.FN
        recall = self.conf_matrix.TPR
        adjusted_fn = original_fn + (1 - recall) * np.sum(self.conf_matrix.y_true() == 0) * self._positive_proportion
        return adjusted_fn

    def get_recall(self):
        adjusted_tp = self.__get_true_positive__()
        adjusted_fn = self.__get_false_negative__()
        return adjusted_tp / (adjusted_tp + adjusted_fn)

    def get_precision(self):
        adjusted_tp = self.__get_true_positive__()
        #adjusted_predicted_positive = np.sum(self.conf_matrix.y_pred()) + self._positive_proportion *  np.sum(self.conf_matrix.y_pred() == 0)
        adjusted_fp = self.__get_false_positive__()
        #print("False positive check: {0} : {1}".format(adjusted_predicted_positive - adjusted_tp, adjusted_fp))
        return adjusted_tp / (adjusted_fp + adjusted_tp)

    def get_f_measure(self, recall, precision):
        if recall == 0 or precision == 0:
            return 0
        return 2 / (1 / recall + 1/precision)