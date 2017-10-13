from models.class_metrics_model import class_metrics_model

model = class_metrics_model()
model.run_train_test_validation()
model.run_train_test_validation_with_cv()