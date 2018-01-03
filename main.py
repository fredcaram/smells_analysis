import time
from models.class_metrics_model import class_metrics_model
from models.method_based_model import method_based_model
from models.history_based_model import history_based_model
import numpy as np

np.random.seed(42)

start = time.time()

model = method_based_model()
#model.run_train_test_validation()
#model.run_balanced_classifier_cv()
model.run_cv_validation()
#model.run_random_search_cv()

#model = method_based_model()
#model.dataset_ids = [2]
#model.run_train_test_validation()
#model.run_cv_validation()

model = class_metrics_model()
#model.run_train_test_validation()
#model.run_balanced_classifier_cv()
model.run_cv_validation()
#model.run_random_search_cv()

#model = class_metrics_model()
#model.dataset_ids = [2]
#model.run_train_test_validation()
#model.run_train_test_validation_with_cv()

model = history_based_model()
#model.run_train_test_validation()
#model.run_balanced_classifier_cv()
model.run_cv_validation()
#model.run_random_search_cv()

end = time.time()
print("Spent time")
print(end - start)