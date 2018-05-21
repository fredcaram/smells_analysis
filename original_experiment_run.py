import os

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from original_experiment import OriginalExperiment


def original_experiment_replication():
    print("****LONG METHOD*****")
    lm_arff = OriginalExperiment.read_arff(os.path.join(".", "original_experiment_dataset", "long-method.arff"))
    lm_cols = lm_arff.columns.values
    lm_arff = lm_arff.rename(columns={lm_cols[0]: "id", lm_cols[1]: "project", lm_cols[2]: "package",
                                      lm_cols[3]: "complextype", lm_cols[4]: "method", lm_cols[-1]: "is_smell"})
    lm_arff = lm_arff.fillna(0)
    params_grid = {
        "max_depth": [i for i in range(1, 5)],
        "min_samples_split": [i for i in range(2, 20)],
        "min_samples_leaf": [i for i in range(1, 5)],
        # "max_features": [None, "auto", "log2"],
        "max_leaf_nodes": [i for i in range(2, 10)],
        "min_impurity_decrease": [1 / (10 ** i) for i in range(1, 6)],
        # "min_weight_fraction_leaf": [i/10 for i in range(0, 5)],
    }
    print("********R Model Test**************")
    OriginalExperiment.train_and_tune_r_tree(lm_arff, "lm_bdt_r")
    # try:
    #     OriginalExperiment.train_and_tune_weka_classifier("", "", lm_arff)
    # finally:
    #     jvm.stop()
    # X_data, y_data = OriginalExperiment.get_x_and_y_from_data(lm_arff)
    # y_pred = np.logical_and(X_data["LOC_method@NUMERIC"] >= 0.8, X_data["CYCLO_method@NUMERIC"] >= 10)
    # OriginalExperiment.print_score(y_pred, y_data, True)
    # Test decision tree
    print("****DECISION TREE TEST*****")
    classifier = DecisionTreeClassifier(random_state=42)
    modelname = "lm_dt"
    gridcv = OriginalExperiment.load_model(modelname)
    if gridcv is None:
        gridcv = OriginalExperiment.tune_classifier(classifier, params_grid, lm_arff)
        OriginalExperiment.save_model(modelname, gridcv)
    print(gridcv.best_estimator_.get_params())
    print(gridcv.best_score_)
    # classifier.set_params(**params)
    # OriginalExperiment.model_cross_validate(best_estimator, lm_arff)
    # Test Random Forest
    print("****RANDOM FOREST TEST*****")
    classifier = RandomForestClassifier(random_state=42)
    # params_grid["n_estimators"] = [i * 2 for i in range(5, 15)]
    modelname = "lm_rf"
    gridcv = OriginalExperiment.load_model(modelname)
    if gridcv is None:
        gridcv = OriginalExperiment.tune_classifier(classifier, params_grid, lm_arff)
        OriginalExperiment.save_model(modelname, gridcv)
    print(gridcv.best_estimator_.get_params())
    print(gridcv.best_score_)
    # Test boosting tree
    print("****ADABOOST DECISION TREE TEST*****")
    classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),
                                    random_state=42)  # GradientBoostingClassifier(random_state=42)
    params_grid = {
        "base_estimator__max_depth": [i for i in range(1, 5)],
        "base_estimator__min_samples_split": [i for i in range(2, 20)],
        "base_estimator__min_samples_leaf": [i for i in range(1, 5)],
        # "base_estimator__max_features": [None, "auto", "log2"],
        "base_estimator__max_leaf_nodes": [i for i in range(2, 10)],
        "base_estimator__min_impurity_decrease": [1 / (10 ** i) for i in range(1, 6)],
        # "min_weight_fraction_leaf": [i/10 for i in range(0, 5)],
    }
    # print(classifier.get_params())
    modelname = "lm_bdt"
    gridcv = OriginalExperiment.load_model(modelname)
    if gridcv is None:
        gridcv = OriginalExperiment.tune_classifier(classifier, params_grid, lm_arff)
        OriginalExperiment.save_model(modelname, gridcv)
    print(gridcv.best_estimator_.get_params())
    print(gridcv.best_score_)
    # Test Naive Bayes
    print("****NAIVE BAYES TEST*****")
    classifier = GaussianNB()
    # params_grid["n_estimators"] = [i * 2 for i in range(5, 15)]
    modelname = "lm_nb"
    model = OriginalExperiment.load_model(modelname)
    if gridcv is None:
        # model = OriginalExperiment.tune_classifier(classifier, params_grid, lm_arff)
        # OriginalExperiment.save_model(modelname, gridcv)
        OriginalExperiment.model_cross_validate(classifier, lm_arff)
        model = classifier
        OriginalExperiment.save_model(modelname, classifier)
    # print(model.get_params())
    # print(gridcv.best_score_)
    print("****FEATURE ENVY*****")
    fe_arff = OriginalExperiment.read_arff(os.path.join(".", "original_experiment_dataset", "feature-envy.arff"))
    fe_cols = fe_arff.columns.values
    fe_arff = fe_arff.rename(columns={fe_cols[0]: "id", fe_cols[1]: "project", fe_cols[2]: "package",
                                      fe_cols[3]: "complextype", fe_cols[4]: "method", fe_cols[-1]: "is_smell"})
    fe_arff = fe_arff.drop(["ATFD_method@NUMERIC", "ATFD_type@NUMERIC", "FDP_method@NUMERIC"], axis=1)
    fe_arff = fe_arff.fillna(0)
    params_grid = {
        "max_depth": [i for i in range(1, 5)],
        "min_samples_split": [i for i in range(2, 20)],
        "min_samples_leaf": [i for i in range(1, 5)],
        # "max_features": [None, "auto", "log2"],
        "max_leaf_nodes": [i for i in range(2, 10)],
        "min_impurity_decrease": [1 / (10 ** i) for i in range(1, 6)],
        # "min_weight_fraction_leaf": [i/10 for i in range(0, 5)],
    }
    # Test decision tree
    print("****DECISION TREE TEST*****")
    classifier = DecisionTreeClassifier(random_state=42)
    modelname = "fe_dt"
    gridcv = OriginalExperiment.load_model(modelname)
    if gridcv is None:
        gridcv = OriginalExperiment.tune_classifier(classifier, params_grid, fe_arff)
        OriginalExperiment.save_model(modelname, gridcv)
    else:
        X_data = OriginalExperiment.get_x(fe_arff)
        OriginalExperiment.print_feature_importances(X_data.columns.values, gridcv.best_estimator_)
        X_data = None
    print(gridcv.best_estimator_.get_params())
    print(gridcv.best_score_)
    # Test Random Forest
    print("****RANDOM FOREST TEST*****")
    classifier = RandomForestClassifier(random_state=42)
    # params_grid["n_estimators"] = [i * 2 for i in range(5, 15)]
    modelname = "fe_rf"
    gridcv = OriginalExperiment.load_model(modelname)
    if gridcv is None:
        gridcv = OriginalExperiment.tune_classifier(classifier, params_grid, fe_arff)
        OriginalExperiment.save_model(modelname, gridcv)
    else:
        X_data = OriginalExperiment.get_x(fe_arff)
        OriginalExperiment.print_feature_importances(X_data.columns.values, gridcv.best_estimator_)
        X_data = None
    print(gridcv.best_estimator_.get_params())
    print(gridcv.best_score_)