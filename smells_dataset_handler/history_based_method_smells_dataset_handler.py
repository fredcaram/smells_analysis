import re

import pandas as pd

from smells_dataset_handler.base_smells_dataset_handler import base_smells_dataset_handler


class history_based_method_smells_dataset_handler(base_smells_dataset_handler):
    def __init__(self):
        base_smells_dataset_handler.__init__(self)
        self.metrics_dir = "change_history"
        self.handled_smell_types = ['ShotgunSurgery',"DivergentChange"]
        self.file_name = "methodChanges"

    def get_handled_smell_types(self):
        return self.handled_smell_types

    def get_metrics_dataframe(self, prefix):
        file = "{0}/{1}/{2}.csv".format(self.metrics_dir, prefix, self.file_name)

        file_df = pd.read_csv(file, sep=";")

        if prefix.startswith("android"):
            metrics_df = self.handle_android_method_change_file(file_df)
        else:
            metrics_df = self.handle_default_method_change_file(file_df)

        metrics_df.columns = ["commit", "instance"]

        #for col in metrics_df.columns.values:
        #    metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

        #metrics_df.index.names = ["instance"]
        #metrics_df = metrics_df.reset_index()
        metrics_df["instance"] = list([self.extract_class_from_method(method) for method in metrics_df["instance"].values])

        return metrics_df

    def extract_class_from_method(self, method_desc):
        m = re.match("(.*)[.]\w+\(.*\)", method_desc)
        if m is None:
            print("error")
            return method_desc
        class_ = m.group(1)
        return class_


    def convert_smells_list_to_df(self, smells):
        smells_by_type = [{"instance": smell["instance"].replace(';', ''), "smell_type": smell["type"]} for smell in
                          smells]
        smells_df = pd.DataFrame(smells_by_type)
        return smells_df

    def handle_android_method_change_file(self, file_df):
        metrics_df = file_df[file_df["Entity"].values == "METHOD"]
        metrics_df = metrics_df.drop(["Date", "BugFix", "Entity", "Public", "ChangeType"], axis=1, errors="ignore")
        return metrics_df

    def handle_default_method_change_file(self, file_df):
        metrics_df = file_df
        metrics_df = metrics_df.drop(["Entity", "Change"], axis=1, errors="ignore")
        return metrics_df