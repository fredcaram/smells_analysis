import numpy as np
import pandas as pd

from models.class_metrics_model import class_metrics_model
from models.history_based_model import divergent_change_model, shotgun_surgery_model
from models.method_based_model import long_method_model, feature_envy_model
from models.parallel_inheritance_model import parallel_inheritance_model


class ExperimentData:
    def __init__(self, force_ratio=None):
        self.force_ratio = force_ratio
        self.models = {
            "Blob": class_metrics_model(),
            "LongMethod": long_method_model(),
            "FeatureEnvy": feature_envy_model(),
            "ParallelInheritance": parallel_inheritance_model(),
            "DivergentChange": divergent_change_model(),
            "ShotgunSurgery": shotgun_surgery_model()
        }

    def get_project_stats(self, projects, smell):
        statistics = {}
        smells_by_project_id = projects.groupby("project_id").aggregate({smell: "sum", "instance": "count"})
        projects_means = smells_by_project_id[smell] / smells_by_project_id["instance"]
        statistics["Smell"] = smell
        statistics["Projects"] = len(smells_by_project_id)
        statistics["Instance Count"] = np.sum(smells_by_project_id["instance"])
        statistics["Smells"] = np.sum(smells_by_project_id[smell])
        statistics["Unlabeled"] = np.sum(smells_by_project_id["instance"]) - np.sum(smells_by_project_id[smell])
        statistics["Avg Instances By Project"] = statistics["Instance Count"] / statistics["Projects"]
        statistics["Avg Smells By Project"] = statistics["Smells"] / statistics["Projects"]
        statistics["Avg Unlabeled By Project"] = statistics["Unlabeled"] / statistics["Projects"]
        statistics["Avg Ratio By Project"] = np.mean(projects_means)
        statistics["Smells Deviation"] = np.std(projects_means)
        return statistics


    def get_statistical_data(self):
        statistics_df = pd.DataFrame()
        for smell, model in self.models.items():
            projects = model.get_dataset(smell)
            df = self.get_project_stats(projects, smell)
            statistics_df = statistics_df.append(df, ignore_index=True)

        statistics_df.to_csv("statistics.csv")

    def get_experiment_data(self):
        for smell, model in self.models.items():
            if self.force_ratio is not None:
                model.samples_proportion = self.force_ratio
            experiment_df = pd.DataFrame()
            print("Smell: {0}".format(smell))
            for model_name, baseline_clf in model.baseline_models.items():
                print("Model: {0}".format(model_name))
                experiment_df = experiment_df.append(self.execute_model(baseline_clf, model_name, model, smell, False, False), ignore_index=True)
                #experiment_df = experiment_df.append(self.execute_model(baseline_clf, model_name, model, smell, True, False), ignore_index=True)
                #experiment_df = experiment_df.append(self.execute_model(baseline_clf, model_name, model, smell, False, True), ignore_index=True)
                #experiment_df = experiment_df.append(self.execute_model(baseline_clf, model_name, model, smell, True, True), ignore_index=True)

            # for model_name, one_class_clf in model.one_class_classifiers.items():
            #     print("Model: {0}".format(model_name))
            #     experiment_df = experiment_df.append(self.execute_model(one_class_clf, model_name, model, smell, False, False, -1), ignore_index=True)
            #     #experiment_df = experiment_df.append(self.execute_model(one_class_clf, model_name, model, smell, True, False, -1), ignore_index=True)
            #
            # for model_name, boosting_clf in model.boosting_models.items():
            #     print("Model: {0}".format(model_name))
            #     experiment_df = experiment_df.append(self.execute_model(boosting_clf, model_name, model, smell, False, False), ignore_index=True)
            #     experiment_df = experiment_df.append(self.execute_model(boosting_clf, model_name, model, smell, True, False), ignore_index=True)
            #     experiment_df = experiment_df.append(self.execute_model(boosting_clf, model_name, model, smell, False, True), ignore_index=True)
            #     experiment_df = experiment_df.append(self.execute_model(boosting_clf, model_name, model, smell, True, True), ignore_index=True)
            #
            # for model_name, ensemble_clf in model.emsemble_models.items():
            #     print("Model: {0}".format(model_name))
            #     experiment_df = experiment_df.append(self.execute_model(ensemble_clf, model_name, model, smell, False, False), ignore_index=True)
            #     experiment_df = experiment_df.append(self.execute_model(ensemble_clf, model_name, model, smell, True, False), ignore_index=True)
            #     experiment_df = experiment_df.append(self.execute_model(ensemble_clf, model_name, model, smell, False, True), ignore_index=True)
            #     experiment_df = experiment_df.append(self.execute_model(ensemble_clf, model_name, model, smell, True, True), ignore_index=True)

            if self.force_ratio is None:
                experiment_df.to_csv("experiment_results_{0}.csv".format(smell))
            else:
                experiment_df.to_csv("experiment_results_withratio_{0}.csv".format(smell))

    def execute_model(self, classifier, model_name, model, smell, use_smotenn, use_puadapter, negative_class=0):
        model_df = {}
        model_df["model"] = model_name
        model_df["smell"] = smell
        model_df["use_smotenn"] = use_smotenn
        model_df["use_puadapter"] = use_puadapter
        model.classifier = classifier
        model.negative_class = negative_class
        model.pu_adapter_enabled = use_puadapter

        if self.force_ratio is None:
            score, pu_score = model.run_cv_validation()
        else:
            score, pu_score = model.run_cv_validation_with_simulated_ratio()

        model_df["precision"] = score[0]
        model_df["recall"] = score[1]
        model_df["fmeasure"] = score[2]
        for k, v in pu_score.items():
            model_df["pu_precision_{0}".format(k)] = v[0]
            model_df["pu_recall_{0}".format(k)] = v[1]
            model_df["pu_fmeasure_{0}".format(k)] = v[2]
        return model_df