import os

import pandas as pd

from repositories.metrics_repository.base_metrics_repository import base_metrics_repository


class method_metrics_repository(base_metrics_repository):
    def __init__(self):
        base_metrics_repository.__init__(self)
        self.metrics_dir = "metrics_files"
        self.metrics_reloaded_class_metrics = ["method_complexity"]

    def get_metrics_dataframe(self, prefix):
        metrics_df = pd.DataFrame()

        for metric in self.metrics_reloaded_class_metrics:
            file = "{0}/{1}_{2}.csv".format(self.metrics_dir, prefix, metric)

            if not os.path.isfile(file):
                continue

            df = pd.read_csv(file, skiprows=1, index_col=0, prefix=metric)

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