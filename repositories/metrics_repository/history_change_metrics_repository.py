import pandas as pd

from repositories.metrics_repository.base_metrics_repository import base_metrics_repository
from repositories.metrics_repository.metrics_repository_helper import extract_class_from_method


class history_change_metrics_repository(base_metrics_repository):
    def __init__(self):
        base_metrics_repository.__init__(self)
        self.metrics_dir = "change_history"
        self.handled_smell_types = ['ShotgunSurgery', "DivergentChange"]
        self.file_name = "methodChanges"

    def get_metrics_dataframe(self, prefix):
        file = "{0}/{1}/{2}.csv".format(self.metrics_dir, prefix, self.file_name)

        file_df = pd.read_csv(file, sep=";")

        if prefix.startswith("android"):
            metrics_df = self.handle_android_method_change_file(file_df)
        else:
            metrics_df = self.handle_default_method_change_file(file_df)

        metrics_df.columns = ["commit", "instance"]

        metrics_df["instance"] = list([extract_class_from_method(method) for method in metrics_df["instance"].values])

        return metrics_df

    def handle_android_method_change_file(self, file_df):
        metrics_df = file_df[file_df["Entity"].values == "METHOD"]
        metrics_df = metrics_df.drop(["Date", "BugFix", "Entity", "Public", "ChangeType"], axis=1, errors="ignore")
        return metrics_df

    def handle_default_method_change_file(self, file_df):
        metrics_df = file_df
        metrics_df = metrics_df.drop(["Entity", "Change"], axis=1, errors="ignore")
        return metrics_df