import time
from models.class_metrics_model import class_metrics_model
from models.method_based_model import method_based_model
from models.history_based_model import history_based_model

start = time.time()

model = method_based_model()
model.run_train_test_validation()
#model.run_train_test_validation_with_cv()

model = class_metrics_model()
model.run_train_test_validation()
#model.run_train_test_validation_with_cv()

model = history_based_model()
model.run_train_test_validation()
#model.run_train_test_validation_with_cv()

end = time.time()
print("Spent time")
print(end - start)