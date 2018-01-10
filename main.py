import time
from models.class_metrics_model import class_metrics_model
from models.method_based_model import long_method_model, feature_envy_model
from models.history_based_model import divergent_change_model, shotgun_surgery_model
import numpy as np

np.random.seed(42)

start = time.time()

model = long_method_model()
#model.run_train_test_validation()
#model.run_balanced_classifier_cv()
#model.run_cv_validation()
#model.run_random_search_cv()

model = feature_envy_model()
#model.run_train_test_validation()
#model.run_balanced_classifier_cv()
#model.run_cv_validation()
#model.run_random_search_cv()

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
model.run_cv_validation()
#model.run_random_search_cv()

end = time.time()
print("Spent time")
print(end - start)