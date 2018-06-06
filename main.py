import time


#jvm.start(system_cp=True, packages=True)


import numpy as np

from original_experiment_run import original_experiment_replication
from experiment_data import ExperimentData
from models.class_metrics_model import class_metrics_model

def current_experiment():
    #pass
    exp_data = ExperimentData()
    exp_data.get_hard_threshold_data()
    exp_data.get_hard_threshold_data_for_positive_only()
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
    model = class_metrics_model()
    model.run_cv_validation()
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