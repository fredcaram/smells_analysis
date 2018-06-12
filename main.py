import os
import time


#jvm.start(system_cp=True, packages=True)


import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold

import original_experiment
from models.model_base import model_base
from original_experiment import OriginalExperiment
from original_experiment_run import original_experiment_replication
from experiment_data import ExperimentData
from models.class_metrics_model import class_metrics_model
from models.method_based_model import long_method_model, feature_envy_model

def current_experiment():
    #pass
    exp_data = ExperimentData()
    exp_data.get_hard_threshold_data()
    #exp_data.get_hard_threshold_data_for_positive_only()
    # exp_data.get_statistical_data()
    #exp_data.get_experiment_data()
    # model = long_method_model()
    # model.run_train_test_validation()
    # model.run_balanced_classifier_cv()
    # scores, pu_scores = model.run_cv_validation()
    # print("scores")
    # print(scores)
    # print("pu_scores")
    # print(pu_scores)
    # model.run_random_search_cv()

    model = long_method_model()
    clf, prf, pus = model.run_cv_validation()
    lm_arff = OriginalExperiment.read_arff(os.path.join(".", "original_experiment_dataset", "long-method.arff"))
    lm_cols = lm_arff.columns.values
    lm_arff = lm_arff.rename(columns={lm_cols[0]: "id", lm_cols[1]: "project", lm_cols[2]: "package",
                                      lm_cols[3]: "complextype", lm_cols[4]: "method", lm_cols[-1]: "is_smell"})
    lm_arff = lm_arff.fillna(0)
    x_data = original_experiment.OriginalExperiment.get_x(lm_arff)
    x_data = original_experiment.OriginalExperiment.standardize_columns(x_data)
    y_data = lm_arff["is_smell"].values.astype(bool).astype(int)

    y_pred = cross_val_predict(clf, x_data, y_data, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    original_experiment.OriginalExperiment.print_score(y_data, y_pred, True)


    model = feature_envy_model()
    clf, prf, pus = model.run_cv_validation()
    fe_arff = OriginalExperiment.read_arff(os.path.join(".", "original_experiment_dataset", "feature-envy.arff"))
    fe_cols = fe_arff.columns.values
    fe_arff = fe_arff.rename(columns={fe_cols[0]: "id", fe_cols[1]: "project", fe_cols[2]: "package",
                                      fe_cols[3]: "complextype", fe_cols[4]: "method", fe_cols[-1]: "is_smell"})
    fe_arff = fe_arff.drop(["ATFD_method@NUMERIC", "ATFD_type@NUMERIC", "FDP_method@NUMERIC"], axis=1)
    fe_arff = fe_arff.fillna(0)
    x_data = original_experiment.OriginalExperiment.get_x(fe_arff)
    x_data = original_experiment.OriginalExperiment.standardize_columns(x_data)
    y_data = (fe_arff["is_smell"].values == "true").astype(int)

    y_pred = cross_val_predict(clf, x_data, y_data, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    original_experiment.OriginalExperiment.print_score(y_data, y_pred, True)
    # model.run_balanced_classifier_cv()
    # model.run_cv_validation()
    # model.run_random_search_cv()
    # model.optimize_ensemble_with_swarm()


if __name__ == '__main__':
    # do stuff with imports and functions defined about
    np.random.seed(42)

    start = time.time()

    current_experiment()

    #original_experiment_replication()


    end = time.time()
    print("Spent time")
    print(end - start)