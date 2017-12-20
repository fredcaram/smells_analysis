import pandas as pd
import re

from repositories.metrics_repository.method_metrics_repository import method_metrics_repository
from repositories.smells_repository.base_smells_repository import base_smells_repository
from repositories.metrics_repository.class_metrics_repository import class_metrics_repository
from repositories.metrics_repository.metrics_repository_helper import extract_class_from_method


class method_smells_repository(base_smells_repository):
    def __init__(self):
        base_smells_repository.__init__(self)
        self.handled_smell_types = ["LongMethod", "FeatureEnvy"]
        self.metrics_repository = method_metrics_repository()
        self.ck_metrics_repository = class_metrics_repository()
        self.ck_metrics_repository.metrics_reloaded_class_metrics = ["ck"]


    def get_handled_smell_types(self):
        return self.handled_smell_types


    def get_metrics_dataframe(self, prefix, dataset_id):
        method_metrics_df = self.metrics_repository.get_metrics_dataframe(prefix, dataset_id)
        if len(method_metrics_df) == 0:
            return method_metrics_df

        method_metrics_df.loc[:, "instance"] = method_metrics_df["instance"].apply(lambda m: m.replace(";", ""))
        method_metrics_df["class_instance"] = method_metrics_df.loc[:, "instance"].apply(lambda m: extract_class_from_method(m))

        #Long method has class instead of method
        if dataset_id == 2:
            method_metrics_df["instance"] = method_metrics_df["class_instance"]

        ckmetrics_df = self.ck_metrics_repository.get_metrics_dataframe(prefix, dataset_id)
        combined_df = method_metrics_df.merge(ckmetrics_df, how="left", left_on="class_instance", right_on="instance", suffixes=("", "_y"))
        combined_df = combined_df.drop(["class_instance", "instance_y"], axis=1)
        return combined_df


    def get_method_part(self, instance):
        regex_match = re.match("(.+;).+", instance)
        if regex_match is None:
            method = instance
        else:
            method = regex_match.group(1)

        #Remove o .java e o que estiver na frente
        #method = re.sub("\.java\..*", "", method)
        # Removetudo que houver após o ultimo ponto antes do parentesis
        #method = re.sub("\.[^.]*\(.*", "", method)

        return method.replace(";", "").replace(" ", "").replace('.java', '')


    def convert_smells_list_to_df(self, smells):
        smells_by_type = [{"instance": self.get_method_part(smell["instance"]), "smell_type": smell["type"]} for smell in
                          smells]
        smells_df = pd.DataFrame(smells_by_type)
        return smells_df
