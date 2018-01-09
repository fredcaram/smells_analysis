from sklearn.metrics import recall_score
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve


class PUScorer(object):
    def __init__(self, positive_proportion, y_true, y_pred):
        self._positive_proportion = positive_proportion
        self.y_true = y_true
        self.y_pred = y_pred
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(y_true, y_pred).ravel()
        self._recall = recall_score(y_true, y_pred, 1)
        self._fpr = self.__get_fpr__()

    def __get_fpr__(self):
        if self.fp == 0:
            return 0

        return self.fp / (self.fp + self.tn)

    def __get_true_positive__(self):
        original_tp = self.tp
        recall = self._recall
        if recall == 0:
            return 0

        adjusted_tp = original_tp + self._positive_proportion * np.sum(self.y_true == 0) * recall
        #adjusted_tp = self._positive_proportion * np.sum(self.conf_matrix.y_true() == 0) * recall
        return adjusted_tp

    def __get_false_positive__(self):
        fpr = self._fpr
        if fpr == 0:
            return 0

        # This yields the same result as the one above
        adjusted_fp = np.sum(self.y_true == 0) * self._positive_proportion * (1 - fpr)
        return adjusted_fp

    def __get_false_negative__(self):
        original_fn = self.fn
        recall = self._recall
        if recall == 0:
            return 0

        adjusted_fn = original_fn +  self._positive_proportion * (1 - recall) * np.sum(self.y_true == 0)
        #adjusted_fn = (1 - recall) * np.sum(self.conf_matrix.y_true() == 0) * self._positive_proportion
        return adjusted_fn

    def get_recall(self):
        adjusted_tp = self.__get_true_positive__()
        adjusted_fn = self.__get_false_negative__()
        if adjusted_tp == 0:
            return 0

        return adjusted_tp / (adjusted_tp + adjusted_fn)

    def get_precision(self):
        adjusted_tp = self.__get_true_positive__()
        #adjusted_predicted_positive = np.sum(self.conf_matrix.y_pred()) + self._positive_proportion *  np.sum(self.conf_matrix.y_pred() == 0)
        adjusted_fp = self.__get_false_positive__()
        if adjusted_tp == 0:
            return 0

        #print("False positive check: {0} : {1}".format(adjusted_predicted_positive - adjusted_tp, adjusted_fp))
        return adjusted_tp / (adjusted_fp + adjusted_tp)

    def get_f_measure(self, recall, precision):
        if recall == 0 or precision == 0:
            return 0
        return 2 / (1 / recall + 1/precision)