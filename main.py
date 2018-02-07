import time

import numpy as np

from experiment_data import ExperimentData
from models.class_metrics_model import class_metrics_model
from models.history_based_model import divergent_change_model, shotgun_surgery_model
from models.method_based_model import long_method_model, feature_envy_model
from models.parallel_inheritance_model import parallel_inheritance_model

np.random.seed(42)

start = time.time()

exp_data = ExperimentData()
#exp_data.get_statistical_data()
exp_data.get_experiment_data()

model = long_method_model()
#model.run_train_test_validation()
#model.run_balanced_classifier_cv()
#scores, pu_scores = model.run_cv_validation()
# print("scores")
# print(scores)
# print("pu_scores")
# print(pu_scores)
#model.run_random_search_cv()

model = feature_envy_model()
#model.run_train_test_validation()
#model.run_balanced_classifier_cv()
#model.run_cv_validation()
#model.run_random_search_cv()
#model.optimize_ensemble_with_swarm()

model = class_metrics_model()
#model.run_train_test_validation()
#model.run_balanced_classifier_cv()
#model.run_cv_validation()
#model.run_random_search_cv()

model = shotgun_surgery_model()
#model.run_train_test_validation()
#model.run_balanced_classifier_cv()
#model.run_cv_validation()
#model.run_random_search_cv()

model = divergent_change_model()
#model.run_train_test_validation()
#model.run_balanced_classifier_cv()
#model.run_cv_validation()
#model.run_random_search_cv()

model = parallel_inheritance_model()
#model.run_train_test_validation()
#model.run_balanced_classifier_cv()
#model.run_cv_validation()
#model.run_random_search_cv()

end = time.time()
print("Spent time")
print(end - start)