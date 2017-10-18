import os

import pandas as pd

from smells_dataset_handler.base_smells_dataset_handler import base_smells_dataset_handler


class metric_reloaded_class_smells_dataset_handler(base_smells_dataset_handler):
    def __init__(self):
        base_smells_dataset_handler.__init__(self)
        self.handled_smell_types = ["Blob"]
        self.metrics_reloaded_class_metrics = ["ck", "class_complexity", "class_dep"]


    def get_handled_smell_types(self):
        return self.handled_smell_types


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


    def convert_smells_list_to_df(self, smells):
        smells_by_type = [{"instance": smell["instance"].replace(';', ''), "smell_type": smell["type"]} for smell in
                          smells]
        smells_df = pd.DataFrame(smells_by_type)
        return smells_df


