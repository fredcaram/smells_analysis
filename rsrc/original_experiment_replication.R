library(caret)

boosted_pruned_dtree_gridcv <- function(X, y, model_name, seed){
  set.seed(seed)
  model_file <- paste0(model_name, ".rds");
  if(!file.exists(model_file)){
    # prepare training scheme
    tr_control <- trainControl(method="repeatedcv", number=10, repeats=1, savePredictions="final");
    # train the model
    y_fac <- factor(c(y));
    model <- train(x = X, y = y_fac, method = "AdaBag", metric = "Accuracy", trControl = tr_control, tuneLength = 10);
    saveRDS(model, model_file);
  }
  #warnings()
  model <- readRDS(model_file);
  
  # Produce confusion matrix from prediction and data used for training
  cf <- confusionMatrix(model$pred$pred, model$pred$obs, mode = "everything");
  print(cf)
  boosted_pruned_dtree_gridcv <- model;
}
