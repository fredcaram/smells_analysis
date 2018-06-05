import os

import pandas as pd

from repositories.metrics_repository.base_metrics_repository import base_metrics_repository


class method_metrics_repository(base_metrics_repository):
    def __init__(self):
        base_metrics_repository.__init__(self)
        self.metrics_dir = "metrics_files"
        self.metrics_reloaded_class_metrics = ["method"]

    def get_metrics_dataframe(self, prefix, dataset_id):
        metrics_df = pd.DataFrame()
        dataset_folder = "dataset_{0}".format(dataset_id)

        for metric in self.metrics_reloaded_class_metrics:
            file = "{0}/{1}/{2}_{3}.csv".format(self.metrics_dir, dataset_folder, prefix, metric)

            if not os.path.isfile(file):
                continue

            df = pd.read_csv(file, skiprows=0, index_col=0, prefix=metric)

            if len(df) == 0:
                continue
            if len(metrics_df) > 0:
                metrics_df = metrics_df.merge(df, left_index=True, right_index=True)
            else:
                metrics_df = df

        for col in metrics_df.columns.values:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

        metrics_df.index.names = ["instance"]
        metrics_df = metrics_df.reset_index()

        return metrics_df